# coding=utf-8
# Megalodon: decoder-only language model implementation (pure PyTorch)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MegalodonConfig:
    """
    Configuration for the Megalodon model.
    """

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=512,
        num_hidden_layers=6,
        intermediate_size=2048,
        ema_projection_size=64,
        bidirectional=False,  # False for decoder (unidirectional EMA)
        shared_representation_size=128,
        use_chunking=True,
        chunk_size=512,
        truncation=None,
        normalize_before_mega=True,
        normalization_type="scalenorm",  # "scalenorm", "layernorm", "rmsnorm", etc.
        norm_affine=True,
        activation="silu",  # Activation in FFN (e.g. "silu", "relu")
        attention_activation="softmax",  # Attention score activation: "softmax", "laplace", or "relu2"
        dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        use_feature_dropout=False,  # if True, use feature-wise dropout (dropping entire feature maps)
        use_normalized_ffn=True,
        nffn_hidden_size=None,
        normalize_before_ffn=True,
        nffn_activation_dropout_prob=0.1,
        max_positions=2048,
        add_token_type_embeddings=False,
        type_vocab_size=2,
        initializer_range=0.02,
        ema_delta_alpha_range=0.2,  # init std for EMA delta/alpha
        ema_beta_range=0.02,  # init std for EMA beta
        ema_gamma_omega_range=1.0,  # init std for EMA gamma/omega
        add_lm_hidden_dense_layer=True,
        is_decoder=True,
        use_cache=True,
        pad_token_id=0,
    ):
        # Model dimensions
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.ema_projection_size = ema_projection_size  # "ndim" for EMA
        self.bidirectional = bidirectional
        self.shared_representation_size = shared_representation_size
        # Attention chunking for long sequences
        self.use_chunking = use_chunking
        self.chunk_size = chunk_size
        self.truncation = truncation
        # Normalization and activation settings
        self.normalize_before_mega = normalize_before_mega
        self.normalization_type = normalization_type
        self.norm_affine = norm_affine
        self.activation = activation
        self.attention_activation = attention_activation
        # Dropout settings
        self.dropout_prob = dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_feature_dropout = use_feature_dropout
        # Feed-forward network
        self.use_normalized_ffn = use_normalized_ffn
        self.nffn_hidden_size = nffn_hidden_size or (
            4 * hidden_size if use_normalized_ffn else None
        )
        self.normalize_before_ffn = normalize_before_ffn
        self.nffn_activation_dropout_prob = nffn_activation_dropout_prob
        # Positional embeddings
        self.max_positions = max_positions
        self.add_token_type_embeddings = add_token_type_embeddings
        self.pad_token_id = pad_token_id
        self.type_vocab_size = type_vocab_size
        # Parameter initialization ranges
        self.initializer_range = initializer_range
        self.ema_delta_alpha_range = ema_delta_alpha_range
        self.ema_beta_range = ema_beta_range
        self.ema_gamma_omega_range = ema_gamma_omega_range
        # Model usage settings
        self.add_lm_hidden_dense_layer = add_lm_hidden_dense_layer
        self.is_decoder = is_decoder
        self.use_cache = use_cache


class MegalodonEmbeddings(nn.Module):
    """
    Token embeddings (plus optional token-type embeddings) for Megalodon.
    """

    def __init__(self, config: MegalodonConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.use_token_types = config.add_token_type_embeddings
        if self.use_token_types:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size
            )
            # Buffer of zeros for token_type_ids if none provided (for tracing)
            self.register_buffer(
                "token_type_ids",
                torch.zeros(config.max_positions, dtype=torch.long).unsqueeze(0),
                persistent=False,
            )
        self.padding_idx = config.pad_token_id

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor = None):
        if input_ids is None:
            raise ValueError("Input ids must be provided for embeddings")
        # Word embeddings
        embeddings = self.word_embeddings(input_ids)
        # Add token type embeddings if applicable
        if self.use_token_types:
            if token_type_ids is None:
                token_type_ids = self.token_type_ids[:, : input_ids.size(1)]
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        return embeddings


class MegalodonSimpleRelativePositionalBias(nn.Module):
    """
    Simple learned relative positional bias for attention.
    """

    def __init__(self, config: MegalodonConfig):
        super().__init__()
        max_rel = config.max_positions if config.chunk_size < 0 else config.chunk_size
        # Bias vector of size (2*max_rel - 1)
        self.rel_bias = nn.Parameter(torch.zeros(2 * max_rel - 1))
        nn.init.normal_(self.rel_bias, mean=0.0, std=config.initializer_range)

    def forward(self, seq_len: int):
        max_rel = (self.rel_bias.size(0) + 1) // 2
        if seq_len > max_rel:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max relative positions {max_rel}"
            )
        # Slice relevant bias range for this sequence length
        start = max_rel - seq_len
        end = start + 2 * seq_len - 1
        bias_slice = self.rel_bias[start:end]  # length = 2*seq_len - 1
        # Construct bias matrix (seq_len x seq_len) from the relative bias vector
        # Position difference i-j maps to index (j - i + seq_len - 1)
        bias_padded = F.pad(bias_slice, (0, seq_len))  # pad to 3*seq_len - 1
        bias_matrix = bias_padded.repeat(seq_len)[: seq_len * (3 * seq_len - 1)]
        bias_matrix = bias_matrix.view(seq_len, 3 * seq_len - 2)
        # Extract central seq_len x seq_len submatrix
        mid = (2 * seq_len - 1) // 2
        bias_matrix = bias_matrix[:, mid : mid + seq_len]
        return bias_matrix


class MegalodonRotaryRelativePositionalBias(nn.Module):
    """
    Rotary positional bias (RoPE-style) with learnable coefficients, as used in Megalodon.
    """

    def __init__(self, config: MegalodonConfig):
        super().__init__()
        if config.hidden_size % 2 != 0:
            raise ValueError("Rotary bias requires even hidden_size")
        self.embed_dim = config.shared_representation_size
        max_pos = config.max_positions if config.chunk_size < 0 else config.chunk_size
        # Precompute sin/cos for maximum positions
        self.sine, self.cosine = self.get_sinusoid_embeddings(max_pos, self.embed_dim)
        # Learned rotation mix coefficients
        self.alpha = nn.Parameter(torch.zeros(1, self.embed_dim))
        self.beta = nn.Parameter(torch.zeros(1, self.embed_dim))
        # Buffer for dtype placement
        self.register_buffer("_float_tensor", torch.FloatTensor(1), persistent=False)

    @staticmethod
    def get_sinusoid_embeddings(max_positions: int, dim: int):
        half = dim // 2
        freq_seq = torch.exp(
            torch.arange(half, dtype=torch.float32) * -(math.log(10000.0) / half)
        )
        t = torch.arange(max_positions, dtype=torch.float32).unsqueeze(
            1
        ) * freq_seq.unsqueeze(0)  # (max_pos, half)
        return torch.sin(t), torch.cos(t)

    def rotary(self, x: torch.Tensor):
        # x: (seq_len, dim)
        seq_len, dim = x.size()
        half = dim // 2
        x1, x2 = torch.chunk(x, 2, dim=-1)
        # Ensure tables cover seq_len
        if seq_len > self.sine.size(0):
            # Recompute if needed
            self.sine, self.cosine = self.get_sinusoid_embeddings(seq_len, dim)
        sin = self.sine[:seq_len].to(x.device)
        cos = self.cosine[:seq_len].to(x.device)
        # Standard RoPE rotation
        rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return rot

    def forward(self, seq_len: int):
        # Compute rotary bias matrices using learned alpha, beta
        rot_alpha = self.rotary(self.alpha.expand(seq_len, self.embed_dim))
        rot_beta = self.rotary(self.beta.expand(seq_len, self.embed_dim))
        # Outer product to form pairwise bias matrix
        bias = torch.einsum("md, nd -> mn", rot_alpha, rot_beta)
        return bias


class MegalodonDropout(nn.Module):
    """
    Unified Dropout module: standard or feature-wise (channel) dropout.
    """

    def __init__(self, p: float, is_featurewise: bool = False):
        super().__init__()
        self.p = p
        self.is_featurewise = is_featurewise

    def forward(self, x: torch.Tensor, batch_first: bool = False):
        if self.is_featurewise:
            # Drop entire features across the sequence
            if batch_first:
                # (batch, seq, features) -> (batch, features, seq) for 2d dropout
                return F.dropout2d(
                    x.transpose(1, 2), p=self.p, training=self.training
                ).transpose(1, 2)
            else:
                if x.dim() != 3:
                    raise ValueError(
                        "Feature dropout expects 3D input (seq_len, batch, features)"
                    )
                # (seq, batch, features) -> (batch, features, seq)
                out = F.dropout2d(x.permute(1, 2, 0), p=self.p, training=self.training)
                return out.permute(2, 0, 1)
        else:
            return F.dropout(x, p=self.p, training=self.training)


class MegalodonRMSNorm(nn.Module):
    """RMS Norm (variation used in Mega/Megalodon)."""

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor):
        # Root-mean-square normalization
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        out = x * rms  # scale by RMS
        if hasattr(self, "weight") and self.weight is not None:
            out = out * self.weight
        return out


class MegalodonScaleNorm(nn.Module):
    """ScaleNorm (scalar normalization)."""

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        if affine:
            self.scalar = nn.Parameter(torch.ones(1))
        else:
            self.register_parameter("scalar", None)

    def forward(self, x: torch.Tensor):
        norm = x.pow(2).mean(dim=self.dim, keepdim=True).add(self.eps).rsqrt()
        out = x * norm
        if hasattr(self, "scalar") and self.scalar is not None:
            out = out * self.scalar
        return out


class MegalodonSequenceNorm(nn.Module):
    """
    Wrapper for sequence normalization (supports multiple norm types).
    """

    def __init__(
        self, norm_type: str, hidden_size: int, eps: float = 1e-5, affine: bool = True
    ):
        super().__init__()
        if norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)
        elif norm_type == "scalenorm":
            self.norm = MegalodonScaleNorm(dim=-1, eps=eps, affine=affine)
        elif norm_type == "rmsnorm":
            self.norm = MegalodonRMSNorm(hidden_size, eps=1e-6, affine=affine)
        elif norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_size, eps=eps, affine=affine)
        elif norm_type == "syncbatchnorm":
            self.norm = nn.SyncBatchNorm(hidden_size, eps=eps, affine=affine)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

    def forward(self, x: torch.Tensor):
        # If using batchnorm, need to reshape [seq, batch, hid] -> [batch, hid, seq]
        if isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            if x.dim() != 3:
                raise ValueError(
                    "BatchNorm requires input of shape (seq_len, batch, hidden)"
                )
            x_bn = self.norm(x.permute(1, 2, 0))
            return x_bn.permute(2, 0, 1)
        else:
            return self.norm(x)


class MegalodonEMA(nn.Module):
    """
    Multi-dimensional Damped Exponential Moving Average (EMA) layer.
    Computes a moving average over sequence positions with learnable decay (alpha) and damping (delta).
    """

    def __init__(self, config: MegalodonConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ndim = config.ema_projection_size
        self.bidirectional = config.bidirectional  # if True, uses two-sided EMA
        self.truncation = config.truncation
        kernel_dim = (
            2 * config.hidden_size if self.bidirectional else config.hidden_size
        )
        # Learnable EMA parameters (will be constrained to [0,1] via sigmoid in use)
        self.damping_factor = nn.Parameter(
            torch.Tensor(kernel_dim, self.ndim, 1)
        )  # corresponds to delta
        self.decay_factor = nn.Parameter(
            torch.Tensor(kernel_dim, self.ndim, 1)
        )  # corresponds to alpha
        self.ema_expansion_matrix = nn.Parameter(
            torch.Tensor(kernel_dim, self.ndim, 1)
        )  # beta parameters
        self.kernel_projection_matrix = nn.Parameter(
            torch.Tensor(kernel_dim, self.ndim)
        )  # gamma parameters
        self.residual_weight = nn.Parameter(
            torch.Tensor(config.hidden_size)
        )  # omega parameter for residual scaling
        # Buffers to cache kernel/coefficient in eval mode
        self.register_buffer("_ema_kernel", None, persistent=False)
        self.register_buffer("_ema_coeffs", None, persistent=False)
        # Parameter initialization (done in model._init_weights)

    def _compute_coeffs(self):
        # Convert damping & decay via sigmoid to (0,1)
        damping = torch.sigmoid(self.damping_factor)
        decay = torch.sigmoid(self.decay_factor)
        prev_w = 1.0 - damping * decay  # weight for previous timestep state
        return damping, prev_w

    def _compute_kernel(self, length: int):
        # Compute EMA convolution kernel of given length (using FFT for efficiency)
        damping, prev_w = self._compute_coeffs()
        L = length
        # Prepare Vandermonde vector: (1,1,L) representing exponents [0..L-1]
        vander = torch.arange(L, device=damping.device, dtype=torch.float32).view(
            1, 1, L
        ) * torch.log(prev_w)
        kernel_vals = (
            damping * self.ema_expansion_matrix * torch.exp(vander)
        )  # shape (kernel_dim, ndim, L)
        # Project through kernel_projection_matrix to get final kernel (kernel_dim x L)
        kernel = torch.einsum(
            "dnl, dn -> dl", kernel_vals, self.kernel_projection_matrix
        ) * math.sqrt(1.0 / self.ndim)
        return kernel  # shape: (kernel_dim, L)

    def get_kernel(self, length: int):
        # Determine effective convolution length (truncate if specified)
        L = length if self.truncation is None else min(self.truncation, length)
        if self.training:
            return self._compute_kernel(L)
        # In eval (inference), cache the kernel to avoid recomputation if sequence length repeats or grows
        if self._ema_kernel is None or self._ema_kernel.size(-1) < L:
            self._ema_kernel = self._compute_kernel(L)
        return self._ema_kernel[..., :L]

    def fft_convolution(
        self, inputs: torch.Tensor, kernel: torch.Tensor, conv_length: int
    ):
        # Perform 1D convolution using FFT (zero-padding to avoid wrap-around)
        # inputs: (batch, hidden_size, seq_len), kernel: (kernel_dim, conv_length)
        # We'll pad to length 2*conv_length for FFT
        inputs_fft = torch.fft.rfft(inputs.float(), n=2 * conv_length)
        kernel_fft = torch.fft.rfft(kernel.float(), n=2 * conv_length)
        conv_result = torch.fft.irfft(inputs_fft * kernel_fft, n=2 * conv_length)
        return conv_result

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        prev_state: torch.Tensor = None,
        use_cache: bool = False,
    ):
        # x shape: (seq_len, batch, hidden_size)
        seq_len, bsz, dim = x.size()
        if dim != self.hidden_size:
            raise ValueError(
                f"EMA input hidden size {dim} != model hidden size {self.hidden_size}"
            )
        # Compute scaled residual connection input
        residual = x * self.residual_weight  # (seq_len, batch, hidden)
        # Prepare input for convolution: (batch, hidden, seq_len)
        x_conv = x.permute(1, 2, 0).contiguous()
        if attention_mask is not None:
            # Mask out padding positions (0 for pads) by zeroing input there
            mask = attention_mask.unsqueeze(1).type_as(x_conv)  # (batch, 1, seq_len)
            x_conv = x_conv * mask
        if self.bidirectional and use_cache:
            raise RuntimeError(
                "Bidirectional EMA does not support incremental decoding"
            )
        if use_cache:
            # Incremental mode: seq_len should be 1
            if seq_len != 1:
                raise ValueError("Incremental EMA expects sequence length 1")
            # Compute one-step EMA update using prev_state
            damping, prev_w = self._compute_coeffs()
            kernel_dim = (
                2 * self.hidden_size if self.bidirectional else self.hidden_size
            )
            # Expand input to shape (batch, kernel_dim, ndim) for elementwise multiply
            # x_conv currently (batch, hidden_size, 1)
            x_expand = x_conv.expand(bsz, kernel_dim, seq_len)  # seq_len=1
            new_state = (
                torch.sigmoid(self.damping_factor) * self.ema_expansion_matrix
            ).squeeze(-1) * x_expand[..., 0]
            if prev_state is not None:
                new_state = (
                    new_state
                    + (
                        1.0
                        - torch.sigmoid(self.damping_factor)
                        * torch.sigmoid(self.decay_factor)
                    ).squeeze(-1)
                    * prev_state
                )
            # Project to output (batch, kernel_dim)
            out_val = torch.einsum(
                "bdn, dn -> bd", new_state, self.kernel_projection_matrix
            ) * math.sqrt(1.0 / self.ndim)
            # If bidirectional, we'd have forward/backward halves in out_val (not used in decoder mode)
            out_val = out_val[:, : self.hidden_size]
            # Apply gating (SiLU) and add residual for output
            out_seq = F.silu(out_val.unsqueeze(0) + residual)  # (1, batch, hidden)
            return out_seq, new_state  # return output (seq_len=1) and updated EMA state
        else:
            # Full sequence mode
            kernel = self.get_kernel(seq_len)  # (kernel_dim, L)
            fft_len = seq_len
            if self.bidirectional:
                # Combine forward/backward EMA kernels
                half_k = kernel.size(0) // 2
                k_forward = kernel[:half_k]  # (hidden, L)
                k_backward = kernel[half_k:]  # (hidden, L)
                # Create full kernel via padding and flipping the backward part:contentReference[oaicite:11]{index=11}
                kernel_full = F.pad(k_forward, (kernel.size(1) - 1, 0)) + F.pad(
                    torch.flip(k_backward, dims=[1]), (0, kernel.size(1) - 1)
                )
                x_conv = F.pad(x_conv, (kernel.size(1) - 1, 0))
                fft_len = seq_len + kernel.size(1) - 1
                kernel_used = kernel_full
                start_index = 2 * kernel.size(1) - 2
            else:
                kernel_used = kernel
                start_index = 0
            # FFT convolution over sequence:contentReference[oaicite:12]{index=12}
            conv_out = self.fft_convolution(x_conv, kernel_used, conv_length=fft_len)[
                ..., start_index : start_index + seq_len
            ]
            conv_out = conv_out.type_as(x_conv)
            # Reshape back to (seq_len, batch, hidden)
            ema_out = conv_out.permute(2, 0, 1)
            # Gated output: SiLU(ema_out + residual)
            out = F.silu(ema_out + residual)
            return out, None


class MegalodonAttention(nn.Module):
    """
    Single-headed gated self-attention with EMA (the Megalodon attention block).
    """

    def __init__(self, config: MegalodonConfig):
        super().__init__()
        self.config = config
        # Activation for value and FFN
        self.act_fn = (
            getattr(F, config.activation) if hasattr(F, config.activation) else F.silu
        )
        # Scale for softmax attention (if using softmax)
        self.scale = (
            (config.shared_representation_size**-0.5)
            if config.attention_activation == "softmax"
            else None
        )
        # Dropouts
        self.dropout = MegalodonDropout(
            config.dropout_prob, is_featurewise=config.use_feature_dropout
        )
        self.hidden_dropout = MegalodonDropout(
            config.hidden_dropout_prob, is_featurewise=config.use_feature_dropout
        )
        self.attention_dropout = MegalodonDropout(
            config.attention_probs_dropout_prob, is_featurewise=False
        )
        # Normalization (layer norm, scale norm, etc.)
        self.norm = MegalodonSequenceNorm(
            config.normalization_type, config.hidden_size, affine=config.norm_affine
        )
        # EMA layer (for moving average state)
        self.ema_layer = MegalodonEMA(config)
        # Linear projections
        self.v_proj = nn.Linear(
            config.hidden_size, config.intermediate_size
        )  # for value
        # mx_proj: projects EMA output to [residual_gate, (query+key+attn_gate), intermediate_state]
        out_dim = (
            config.hidden_size
            + (config.shared_representation_size + config.intermediate_size)
            + config.hidden_size
        )
        self.mx_proj = nn.Linear(config.hidden_size, out_dim)
        self.h_proj = nn.Linear(
            config.intermediate_size, config.hidden_size
        )  # project attention output to hidden
        # Parameters for generating Q and K from shared representation
        self.qk_weight = nn.Parameter(
            torch.Tensor(2, config.shared_representation_size)
        )
        self.qk_bias = nn.Parameter(torch.Tensor(2, config.shared_representation_size))
        # Relative positional bias module
        if config.normalization_type not in ["simple", "rotary"]:
            # If unspecified, default to rotary in this implementation
            config.relative_positional_bias = "rotary"
        if getattr(config, "relative_positional_bias", "rotary") == "simple":
            self.pos_bias = MegalodonSimpleRelativePositionalBias(config)
        else:
            self.pos_bias = MegalodonRotaryRelativePositionalBias(config)
        # Choose attention activation function
        if config.attention_activation == "softmax":
            self.attn_activation_fn = None  # use softmax directly
        else:
            if config.attention_activation == "laplace":
                # Laplace attention: exp(-|x|)
                self.attn_activation_fn = lambda x: torch.exp(-torch.abs(x))
            elif config.attention_activation == "relu2":
                # ReLU^2 attention
                self.attn_activation_fn = lambda x: F.relu(x) ** 2
            else:
                raise ValueError(
                    f"Unknown attention activation: {config.attention_activation}"
                )
        # Initialize projection parameters
        nn.init.normal_(self.qk_weight, mean=0.0, std=config.initializer_range)
        nn.init.zeros_(self.qk_bias)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None,
        causal_mask: torch.Tensor = None,
        past_key: torch.Tensor = None,
        past_value: torch.Tensor = None,
        past_ema: torch.Tensor = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        # Input: x (seq_len, batch, hidden)
        seq_len, bsz, _ = x.size()
        residual = x  # save residual connection
        if self.config.normalize_before_mega:
            x = self.norm(x)
        # Compute value representation
        value = self.act_fn(self.v_proj(x))  # (seq_len, batch, intermediate_size)
        # EMA output (with optional previous EMA state for cache)
        ema_out, updated_ema_state = self.ema_layer(
            x, attention_mask=padding_mask, prev_state=past_ema, use_cache=use_cache
        )
        ema_out = self.dropout(ema_out)
        # Project EMA output to gates and intermediate state
        combined = self.mx_proj(ema_out)
        # Split combined projection into 3 parts
        res_gate, qk_gate, interm_state = torch.split(
            combined,
            [
                self.config.hidden_size,
                self.config.shared_representation_size + self.config.intermediate_size,
                self.config.hidden_size,
            ],
            dim=-1,
        )
        res_gate = torch.sigmoid(res_gate)  # residual gate (sigmoid)
        qk_gate = F.silu(qk_gate)  # SiLU for query/key and attention gate
        # Split into query+key shared rep and attention output gate
        qk_shared, attn_gate = torch.split(
            qk_gate,
            [self.config.shared_representation_size, self.config.intermediate_size],
            dim=-1,
        )
        # Compute queries and keys from shared rep: shape (seq_len, batch, 2, shared_repr)
        qk_combined = (
            qk_shared.unsqueeze(2) * self.qk_weight + self.qk_bias
        )  # apply linear transform
        query, key = torch.unbind(
            qk_combined, dim=2
        )  # each (seq_len, batch, shared_repr)
        # Transpose to (batch, seq_len, dim) for attention computation
        query = query.transpose(0, 1)  # (batch, seq_len, shared_repr)
        key = key.transpose(0, 1)  # (batch, seq_len, shared_repr)
        value_bf = value.transpose(0, 1)  # (batch, seq_len, intermediate)
        # Append past key/value states if using cache (incremental decoding)
        if use_cache and past_key is not None and past_value is not None:
            key = torch.cat([past_key, key], dim=1)
            value_bf = torch.cat([past_value, value_bf], dim=1)
        # Determine updated key/value to return for cache
        if not self.config.use_chunking or (key.size(1) % self.config.chunk_size != 0):
            updated_key = key
            updated_value = value_bf
        else:
            # If chunk boundary reached exactly, do not carry over (truncate history):contentReference[oaicite:13]{index=13}
            updated_key = None
            updated_value = None
        # Prepare chunking: reshape sequences into chunks if enabled
        if not self.config.use_chunking:
            # No chunking: add a singleton chunk dimension
            query = query.unsqueeze(1)  # (batch, 1, seq_len, shared_repr)
            key = key.unsqueeze(1)
            value_bf = value_bf.unsqueeze(1)
            if padding_mask is not None:
                padding_mask = padding_mask.unsqueeze(1)  # (batch, 1, seq_len)
        else:
            seq = query.size(1)
            ctx = key.size(1)
            # Split target sequence into n_chunks
            if seq < self.config.chunk_size:
                query = query.unsqueeze(1)
            else:
                n_chunks_q = seq // self.config.chunk_size
                query = query.reshape(
                    bsz,
                    n_chunks_q,
                    self.config.chunk_size,
                    self.config.shared_representation_size,
                )
            # Split context (keys/values) sequence similarly
            if ctx < self.config.chunk_size:
                key = key.unsqueeze(1)
                value_bf = value_bf.unsqueeze(1)
                if padding_mask is not None:
                    padding_mask = padding_mask.unsqueeze(1)
            else:
                n_chunks_k = ctx // self.config.chunk_size
                key = key.reshape(
                    bsz,
                    n_chunks_k,
                    self.config.chunk_size,
                    self.config.shared_representation_size,
                )
                value_bf = value_bf.reshape(
                    bsz,
                    n_chunks_k,
                    self.config.chunk_size,
                    self.config.intermediate_size,
                )
                if padding_mask is not None:
                    padding_mask = padding_mask.view(
                        bsz, n_chunks_k, self.config.chunk_size
                    )
        # Compute attention scores with relative positional bias
        # bias shape: (chunk_size, chunk_size) or (seq_len, seq_len):contentReference[oaicite:14]{index=14}
        bias = self.pos_bias(key.size(-2) if self.config.use_chunking else key.size(1))
        if self.config.attention_activation == "softmax":
            # Softmax attention (with additive masks)
            attn_scores = (
                torch.matmul(query, key.transpose(-2, -1)) + bias
            )  # (batch, n_chunks, L, L)
            if causal_mask is not None:
                # Convert causal mask (1/0) to additive form (0 or -inf)
                # causal_mask: shape (L, L) where L = chunk_size or seq_len
                mask = causal_mask == 0
                attn_scores = attn_scores.masked_fill(mask, float("-inf"))
            if padding_mask is not None:
                # padding_mask: 1 for real tokens, 0 for pads
                # Invert to get mask for pads
                inv_pad = (1 - padding_mask).bool()  # True where pad
                attn_scores = attn_scores.masked_fill(
                    inv_pad.unsqueeze(-2), float("-inf")
                )
            attn_weights = F.softmax(attn_scores, dim=-1)  # softmax over last dim
        else:
            # Elementwise attention (Laplace or ReLU^2)
            lengths = key.size(
                -2
            )  # effective context length per chunk (or total seq_len if no chunk)
            # Normalize scores by sequence length (as in original for relu^2/laplace):contentReference[oaicite:15]{index=15}
            attn_scores = torch.matmul(query, key.transpose(-2, -1)) / lengths + bias
            attn_weights = self.attn_activation_fn(attn_scores)
            if padding_mask is not None:
                attn_weights = attn_weights * padding_mask.unsqueeze(-2)
            if causal_mask is not None:
                attn_weights = attn_weights * causal_mask
        attn_weights = self.attention_dropout(attn_weights)
        # Apply attention weights to values
        context = torch.matmul(
            attn_weights, value_bf
        )  # (batch, n_chunks, L, intermediate)
        # Reshape back to sequence form
        if self.config.use_chunking:
            context = context.reshape(
                bsz, -1, self.config.intermediate_size
            )  # (batch, seq_len, intermediate)
        # Transpose back to (seq_len, batch, intermediate)
        context = context.transpose(0, 1)  # (seq_len, batch, intermediate_size)
        # Apply attention output gate and project to hidden size
        attn_output = self.h_proj(context * attn_gate)
        # Combine with intermediate state and apply activation
        combined_out = self.act_fn(interm_state + attn_output)
        combined_out = self.dropout(combined_out)
        # Final output with gated residual addition:contentReference[oaicite:16]{index=16}
        out = residual + res_gate * (combined_out - residual)
        if not self.config.normalize_before_mega:
            out = self.norm(out)
        # Prepare return tuple
        outputs = (out,)
        if output_attentions:
            outputs += (attn_weights.detach() if not self.training else attn_weights,)
        if use_cache:
            outputs += (updated_key, updated_value, updated_ema_state)
        return outputs


class MegalodonFFN(nn.Module):
    """Normalized feed-forward network (NFFN) block for Megalodon."""

    def __init__(self, config: MegalodonConfig):
        super().__init__()
        self.prenorm = config.normalize_before_ffn
        self.norm = MegalodonSequenceNorm(
            config.normalization_type, config.hidden_size, affine=config.norm_affine
        )
        self.fc1 = nn.Linear(config.hidden_size, config.nffn_hidden_size)
        self.fc2 = nn.Linear(config.nffn_hidden_size, config.hidden_size)
        self.act_fn = (
            getattr(F, config.activation) if hasattr(F, config.activation) else F.silu
        )
        self.dropout = MegalodonDropout(
            config.dropout_prob, is_featurewise=config.use_feature_dropout
        )
        self.hidden_dropout = MegalodonDropout(
            config.nffn_activation_dropout_prob,
            is_featurewise=config.use_feature_dropout,
        )

    def forward(self, x: torch.Tensor):
        residual = x
        if self.prenorm:
            x = self.norm(x)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.hidden_dropout(hidden)
        out = self.fc2(hidden)
        out = self.dropout(out)
        out = out + residual
        if not self.prenorm:
            out = self.norm(out)
        return out


class MegalodonBlock(nn.Module):
    """
    Single decoder block (self-attention + optional FFN).
    """

    def __init__(self, config: MegalodonConfig):
        super().__init__()
        self.attention = MegalodonAttention(config)
        self.ffn = MegalodonFFN(config) if config.use_normalized_ffn else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        causal_mask: torch.Tensor = None,
        past_states: tuple = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        # past_states: tuple of (past_key, past_value, past_ema) from previous step (for caching)
        past_key, past_value, past_ema = (
            past_states if past_states is not None else (None, None, None)
        )
        attn_outputs = self.attention(
            hidden_states,
            padding_mask=attention_mask,
            causal_mask=causal_mask,
            past_key=past_key,
            past_value=past_value,
            past_ema=past_ema,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # attn_outputs: (hidden_states, [attn_weights], [updated_key, updated_value, updated_ema])
        hidden_states = attn_outputs[0]
        attn_weights = attn_outputs[1] if output_attentions else None
        new_key = (
            attn_outputs[2]
            if use_cache and attn_weights is not None
            else (attn_outputs[1] if use_cache else None)
        )
        new_value = (
            attn_outputs[3]
            if use_cache and attn_weights is not None
            else (attn_outputs[2] if use_cache else None)
        )
        new_ema = (
            attn_outputs[4]
            if use_cache and attn_weights is not None
            else (attn_outputs[3] if use_cache else None)
        )
        if self.ffn is not None:
            hidden_states = self.ffn(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += ((new_key, new_value, new_ema),)
        return outputs


class MegalodonModel(nn.Module):
    """
    The bare Megalodon Model outputting raw hidden states.
    """

    def __init__(self, config: MegalodonConfig):
        super().__init__()
        self.config = config
        self.embeddings = MegalodonEmbeddings(config)
        self.layers = nn.ModuleList(
            [MegalodonBlock(config) for _ in range(config.num_hidden_layers)]
        )
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize parameters (following Mega defaults)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight, mean=0.0, std=self.config.initializer_range
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight, mean=0.0, std=self.config.initializer_range
                )
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, MegalodonEMA):
                # EMA parameters (delta, alpha ~ N(0, 0.2); beta ~ N(0, 0.02) + alternating 1,-1; gamma/omega ~ N(0, 1.0))
                nn.init.normal_(
                    module.damping_factor,
                    mean=0.0,
                    std=self.config.ema_delta_alpha_range,
                )
                nn.init.normal_(
                    module.decay_factor, mean=0.0, std=self.config.ema_delta_alpha_range
                )
                nn.init.normal_(
                    module.ema_expansion_matrix,
                    mean=0.0,
                    std=self.config.ema_beta_range,
                )
                # Set beta initial signs: [1, -1, 1, -1, ...] for stability:contentReference[oaicite:17]{index=17}
                if module.ema_expansion_matrix.size(0) > 0:
                    # Alternate sign on expansion matrix first dimension
                    idx = torch.arange(module.ema_expansion_matrix.size(0))
                    module.ema_expansion_matrix.data[:, :, 0] += torch.where(
                        idx % 2 == 0, 1.0, -1.0
                    ).unsqueeze(1)
                nn.init.normal_(
                    module.kernel_projection_matrix,
                    mean=0.0,
                    std=self.config.ema_gamma_omega_range,
                )
                nn.init.normal_(
                    module.residual_weight,
                    mean=0.0,
                    std=self.config.ema_gamma_omega_range,
                )
            elif isinstance(module, MegalodonRotaryRelativePositionalBias):
                nn.init.normal_(
                    module.alpha, mean=0.0, std=self.config.initializer_range
                )
                nn.init.normal_(
                    module.beta, mean=0.0, std=self.config.initializer_range
                )

    def _build_causal_mask(self, seq_len: int, device):
        # Lower-triangular causal mask (1 = keep, 0 = mask)
        return torch.tril(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.uint8)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        past_key_values: list = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        # Embed tokens
        embedding_output = self.embeddings(input_ids, token_type_ids)
        # Transpose to (seq_len, batch, hidden) for internal processing
        hidden_states = embedding_output.transpose(
            0, 1
        )  # (seq_len, batch, hidden_size)
        seq_len = hidden_states.size(0)
        batch_size = hidden_states.size(1)
        # If chunking is used, ensure sequence length is multiple of chunk_size (for simplicity):contentReference[oaicite:18]{index=18}
        if self.config.use_chunking and seq_len > self.config.chunk_size:
            if seq_len % self.config.chunk_size != 0:
                raise ValueError(
                    f"Sequence length {seq_len} must be a multiple of chunk_size {self.config.chunk_size} when chunking."
                )
        # Prepare padding mask (1 for tokens to attend, 0 for pads)
        if attention_mask is not None:
            # Convert to float mask (same shape)
            padding_mask = attention_mask.to(torch.float32)
        else:
            padding_mask = None
        # Prepare causal mask (lower triangular matrix for seq_len, or chunk_size if chunking):contentReference[oaicite:19]{index=19}
        causal_mask = None
        if self.config.is_decoder:
            L = (
                self.config.chunk_size
                if (self.config.use_chunking and seq_len > self.config.chunk_size)
                else seq_len
            )
            causal_mask = self._build_causal_mask(L, hidden_states.device)
        # Past key values initialization
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        elif len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values length must equal number of layers")
        # Collect outputs
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        new_past_key_values = [] if use_cache else None
        # Forward through layers
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(
                    hidden_states.transpose(0, 1)
                )  # store (batch, seq, hidden)
            layer_past = past_key_values[idx]
            layer_outputs = layer(
                hidden_states,
                attention_mask=padding_mask,
                causal_mask=causal_mask,
                past_states=layer_past,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]  # updated hidden states
            if output_attentions:
                all_attentions.append(layer_outputs[1])
            if use_cache:
                new_past_key_values.append(layer_outputs[-1])
        # Transpose final hidden to (batch, seq, hidden)
        hidden_states = hidden_states.transpose(0, 1)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        outputs = (hidden_states,)
        if output_hidden_states:
            outputs += (all_hidden_states,)
        if output_attentions:
            outputs += (all_attentions,)
        if use_cache:
            outputs += (new_past_key_values,)
        return outputs  # last_hidden_state, (optional: all_hidden_states, all_attentions, new_past_key_values)


class MegalodonForCausalLM(nn.Module):
    """
    Megalodon Model with a language modeling head on top for causal LM tasks.
    """

    def __init__(self, config: MegalodonConfig):
        super().__init__()
        self.config = config
        self.megalodon = MegalodonModel(config)
        # Optional intermediate dense layer before LM head
        if config.add_lm_hidden_dense_layer:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.activation = nn.Tanh()
        else:
            self.dense = None
        # Language modeling head (tied with input embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = (
            self.megalodon.embeddings.word_embeddings.weight
        )  # tie weights

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        labels: torch.LongTensor = None,
        past_key_values: list = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        # Forward through base model
        outputs = self.megalodon(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = outputs[0]  # last hidden states (batch, seq_len, hidden)
        if self.dense is not None:
            hidden_states = self.activation(self.dense(hidden_states))
        logits = self.lm_head(hidden_states)  # (batch, seq_len, vocab_size)
        loss = None
        if labels is not None:
            # Shift prediction scores and labels for causal language modeling loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        # Return tuple: (loss if provided, logits, past_key_values, hidden_states, attentions)
        output_tuple = (logits,)
        # Append optional outputs from base model
        if output_attentions or output_hidden_states or use_cache:
            output_tuple += tuple(outputs[1:])
        if loss is not None:
            output_tuple = (loss,) + output_tuple
        return output_tuple
