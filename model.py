"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import inspect
from dataclasses import dataclass
import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# Debug helper: check for NaN/Inf in tensors (gated by DEBUG_NAN env var)
DEBUG_NAN = os.environ.get('DEBUG_NAN', '0') == '1'
if DEBUG_NAN:
    torch.autograd.set_detect_anomaly(True)

def _check_finite(t, name):
    """Check if tensor contains NaN or Inf values. Only runs if DEBUG_NAN=1."""
    if not DEBUG_NAN:
        return
    if not torch.isfinite(t).all():
        finite_mask = torch.isfinite(t)
        num_bad = (~finite_mask).sum().item()
        total = t.numel()
        finite_vals = t[finite_mask]
        if finite_vals.numel() > 0:
            nan_count = torch.isnan(finite_vals).sum().item()
            inf_count = torch.isinf(finite_vals).sum().item()
            nan_percent = nan_count / finite_vals.numel() * 100
            inf_percent = inf_count / finite_vals.numel() * 100
            mean_val = finite_vals.mean().item()
            min_val = finite_vals.min().item()
            max_val = finite_vals.max().item()
        else:
            mean_val = min_val = max_val = float('nan')
        raise RuntimeError(f"Non-finite values detected in {name}: {num_bad}/{total} non-finite. stats of finite values -> nan={nan_percent}%, inf={inf_percent}%, mean={mean_val}, min={min_val}, max={max_val}")

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # initialize to 1/sqrt(head_size) per common attention scaling
        head_size = config.n_embd // config.n_head
        self.register_buffer("scale", torch.tensor(1.0 / math.sqrt(head_size)))
        # TURNING OFF FLASH ATTENTION FOR NOW
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = False # hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # Debug: check input to attention
        _check_finite(x, "x (input to CausalSelfAttention)")
        
        # Debug: check c_attn weights
        _check_finite(self.c_attn.weight, "c_attn.weight")
        if self.c_attn.bias is not None:
            _check_finite(self.c_attn.bias, "c_attn.bias")

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        _check_finite(qkv, "qkv (after c_attn projection)")
        q, k, v  = qkv.split(self.n_embd, dim=2)
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)

        # normalize key and query vectors before attention
        # q = F.normalize(q, p=2, dim=-1, eps=1e-8)
        # k = F.normalize(k, p=2, dim=-1, eps=1e-8)
        # v = F.normalize(v, p=2, dim=-1, eps=1e-8)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, -torch.inf)
        # att = F.softmax(att, dim=-1)
        # att = self.attn_dropout(att)
        # round att using a straight-through estimator
        def straight_through_estimator(x):
            threshold = 0.1 # selected based on the distribution of the attention weights
            return x + (torch.where(x < threshold, torch.zeros_like(x), x) - x).detach()
        att = F.sigmoid(att)
        if not self.training:
            self.last_att = att.detach()
        att_samples = torch.bernoulli(att)
        att = att + (att_samples - att).detach() # straight through estimator

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # return the sum of the attention values per head
        att_sum = att.sum(dim=-1)
        att_var = (att * (1-att)).sum(dim=-1)
        # assert torch.all((att_var >= 0) & att_sum >= 0), "Attention variance and sum out of range [0, 1]"
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att_sum, att_var

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        vals, att_sum, att_var = self.attn(x)
        x = x + vals
        x = x + self.mlp(self.ln_2(x))
        return x, att_sum, att_var

@dataclass
class GPTConfig:
    # NOTE: this does not actually affect the model, these are dummy values
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def get_attention_scales(self):
        """
        Return a dictionary of attention scale values from all layers.
        Returns dict with keys like 'attn_scale/layer_0', 'attn_scale/layer_1', etc.
        Also includes 'attn_scale/mean' and 'attn_scale/std' for aggregated stats.
        """
        scales = {}
        scale_values = []
        for i, block in enumerate(self.transformer.h):
            scale_val = block.attn.scale.item()
            scales[f'attn_scale/layer_{i}'] = scale_val
            scale_values.append(scale_val)
        
        # Add aggregated statistics
        if scale_values:
            scales['attn_scale/mean'] = sum(scale_values) / len(scale_values)
            scales['attn_scale/std'] = math.sqrt(sum((x - scales['attn_scale/mean'])**2 for x in scale_values) / len(scale_values))
            scales['attn_scale/min'] = min(scale_values)
            scales['attn_scale/max'] = max(scale_values)
        
        return scales

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        _check_finite(tok_emb, "tok_emb (token embeddings)")
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        _check_finite(pos_emb, "pos_emb (position embeddings)")
        x = self.transformer.drop(tok_emb + pos_emb)
        _check_finite(x, "x (after embeddings + dropout, before blocks)")
        att_sums = []
        att_vars = []
        for i, block in enumerate(self.transformer.h):
            x, att_sum, att_var = block(x)
            _check_finite(x, f"x (after block {i})")
            att_sums.append(att_sum)
            att_vars.append(att_var)
            # print(f"block {i}: att_sum: {att_sum.median():.4f}, att_var: {att_var.median():.4f}")
            # print(f"block {i}: att_sum: {att_sum.quantile(0.25):.4f}, att_var: {att_var.quantile(0.25):.4f}")
            # print(f"block {i}: att_sum: {att_sum.quantile(0.75):.4f}, att_var: {att_var.quantile(0.75):.4f}")
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            margin = 100.0 # up to 10 close matches ~1.0

            # Stack attention sums: att_sums: list of (batch, n_head) => (n_layers, batch, n_head)
            att_sums_tensor = torch.vstack(att_sums)  # (n_layers, batch, n_head)
            att_vars_tensor = torch.vstack(att_vars)  # (n_layers, batch, n_head)
            _check_finite(att_sums_tensor, "att_sums_tensor")
            _check_finite(att_vars_tensor, "att_vars_tensor")
            def zeroed_softplus(x, margin, beta):
                return F.softplus(x - margin, beta) - F.softplus(torch.tensor(-margin), beta)
            def approx_gaussian_cdf(x, mu, var):
                # Add epsilon inside sqrt to prevent NaN gradients when var→0
                # sqrt gradient is 1/(2*sqrt(var)), which explodes if var is tiny
                return torch.sigmoid(1.72 * (x - mu) / torch.sqrt(var + 1e-2))
            def hinge_loss(x, margin):
                return torch.max(margin - x, torch.zeros_like(x))
            # Compute sparsity objective
            sparsity_objective = zeroed_softplus(att_sums_tensor, margin, 0.25) # - approx_gaussian_cdf(margin, att_sums_tensor, att_vars_tensor)
            _check_finite(sparsity_objective, "sparsity_objective")
            sparsity_weight = 1e-2
            total_sparsity_objective = sparsity_weight * sparsity_objective.sum() / (b * t * self.config.n_head * self.config.n_layer)
            _check_finite(total_sparsity_objective, "total_sparsity_objective")
            loss_dict = {'ce_loss': ce_loss, 'sparsity_objective': total_sparsity_objective}
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss_dict = None

        return logits, loss_dict

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"number of layers: {self.config.n_layer}")
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
