import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=2048):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )
        pe = torch.zeros(max_len, hidden_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TsAttnBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        return self.norm2(x + self.dropout(ff_out))


class LandscapeEncoder(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.cross_individual = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)
        self.cross_dimension = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)
        self.position = PositionalEncoding(hidden_dim)

    def forward(self, x):
        bsz, n_obj, n_dim, n_ind, hidden_dim = x.shape

        x = x.reshape(bsz * n_obj * n_dim, n_ind, hidden_dim)
        x = self.cross_individual(x)
        x = x.reshape(bsz, n_obj, n_dim, n_ind, hidden_dim)

        x = x.permute(0, 1, 3, 2, 4).reshape(bsz * n_obj * n_ind, n_dim, hidden_dim)
        x = self.position(x)
        x = self.cross_dimension(x)
        x = x.reshape(bsz, n_obj, n_ind, n_dim, hidden_dim)

        return x.mean(dim=3)


class CrossSpaceAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_out, _ = self.attn(query, key, value, need_weights=False)
        x = self.norm1(query + self.dropout(attn_out))
        ff_out = self.ff(x)
        return self.norm2(x + self.dropout(ff_out))


class BaseAgent(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        n_heads=8,
        ff_dim=256,
        dropout=0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.W_true = nn.Linear(2, hidden_dim)
        self.W_surr = nn.Linear(3, hidden_dim)
        self.encoder_true = LandscapeEncoder(hidden_dim, n_heads, ff_dim, dropout)
        self.encoder_surr = LandscapeEncoder(hidden_dim, n_heads, ff_dim, dropout)
        self.cross_space_attn = CrossSpaceAttention(hidden_dim, n_heads, ff_dim, dropout)

    @staticmethod
    def _ensure_batch(x):
        return x.unsqueeze(0) if x.dim() == 2 else x

    @staticmethod
    def _normalize_by_range(x, lower, upper):
        denom = (upper - lower).clamp_min(1e-12)
        return ((x - lower) / denom).clamp(0.0, 1.0)

    @staticmethod
    def _normalize_by_extrema(*tensors):
        stacked = torch.cat(tensors, dim=1)
        min_v = stacked.amin(dim=1, keepdim=True)
        max_v = stacked.amax(dim=1, keepdim=True)
        denom = (max_v - min_v).clamp_min(1e-12)
        return tuple(((tensor - min_v) / denom).clamp(0.0, 1.0) for tensor in tensors)

    @staticmethod
    def _prepare_progress(
        progress,
        device,
        dtype,
        batch_size,
    ):
        progress = torch.as_tensor(
            progress,
            device=device,
            dtype=dtype,
        )

        if progress.dim() == 0:
            progress = progress.repeat(batch_size)

        progress = progress.reshape(batch_size, -1)

        if progress.size(1) != 1:
            progress = progress[:, :1]

        return progress.clamp(0.0, 1.0)

    def _prepare_inputs(
        self,
        x_true,
        y_true,
        x_sur,
        y_sur,
        sigma_sur,
        lower_bound,
        upper_bound,
    ):
        x_true = self._ensure_batch(x_true).float()
        y_true = self._ensure_batch(y_true).float()
        x_sur = self._ensure_batch(x_sur).float()
        y_sur = self._ensure_batch(y_sur).float()
        sigma_sur = self._ensure_batch(sigma_sur).float()

        device = x_true.device
        dtype = x_true.dtype
        lower = torch.as_tensor(lower_bound, device=device, dtype=dtype)
        upper = torch.as_tensor(upper_bound, device=device, dtype=dtype)

        if lower.dim() == 0:
            lower = lower.repeat(x_true.size(-1))
        if upper.dim() == 0:
            upper = upper.repeat(x_true.size(-1))

        lower = lower.view(1, 1, -1)
        upper = upper.view(1, 1, -1)

        x_true = self._normalize_by_range(x_true, lower, upper)
        x_sur = self._normalize_by_range(x_sur, lower, upper)
        y_true, y_sur = self._normalize_by_extrema(y_true, y_sur)
        (sigma_sur,) = self._normalize_by_extrema(sigma_sur)

        x_true_expand = x_true.transpose(1, 2).unsqueeze(1).unsqueeze(-1)
        x_true_expand = x_true_expand.expand(-1, y_true.size(-1), -1, -1, -1)
        y_true_expand = y_true.transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        y_true_expand = y_true_expand.expand(-1, -1, x_true.size(-1), -1, -1)
        m_true = torch.cat((x_true_expand, y_true_expand), dim=-1)

        x_sur_expand = x_sur.transpose(1, 2).unsqueeze(1).unsqueeze(-1)
        x_sur_expand = x_sur_expand.expand(-1, y_sur.size(-1), -1, -1, -1)
        y_sur_expand = y_sur.transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        y_sur_expand = y_sur_expand.expand(-1, -1, x_sur.size(-1), -1, -1)
        sigma_expand = sigma_sur.transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        sigma_expand = sigma_expand.expand(-1, -1, x_sur.size(-1), -1, -1)
        m_surr = torch.cat((x_sur_expand, y_sur_expand, sigma_expand), dim=-1)

        return m_true, m_surr

    def encode(
        self,
        x_true,
        y_true,
        x_sur,
        y_sur,
        sigma_sur,
        progress,
        lower_bound,
        upper_bound,
    ):
        m_true, m_surr = self._prepare_inputs(
            x_true=x_true,
            y_true=y_true,
            x_sur=x_sur,
            y_sur=y_sur,
            sigma_sur=sigma_sur,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        e_true = self.W_true(m_true)
        e_surr = self.W_surr(m_surr)

        s_true = self.encoder_true(e_true)
        s_surr = self.encoder_surr(e_surr)

        h_true = s_true.mean(dim=1)
        h_surr_raw = s_surr.mean(dim=1)
        h_surr = self.cross_space_attn(h_surr_raw, h_true, h_true)

        progress = self._prepare_progress(
            progress=progress,
            device=h_true.device,
            dtype=h_true.dtype,
            batch_size=h_true.size(0),
        )

        return {
            "H_true": h_true,
            "H_surr": h_surr,
            "progress": progress,
        }
