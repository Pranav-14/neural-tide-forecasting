"""TiDE (Time-series Dense Encoder) with RevIN in PyTorch."""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization for time-series forecasting against distribution shifts.
    
    Reference:
      Kim et al., "Reversible Instance Normalization for Accurate Time-Series
      Forecasting against Distribution Shift", ICLR 2022.
    """

    def __init__(self, num_features: int = 1, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        self.mean: Optional[torch.Tensor] = None
        self.stdev: Optional[torch.Tensor] = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor along the time dimension (dim=1).
        
        Args:
            x: Tensor of shape (B, L) or (B, L, D).
        """
        # Ensure 3D shape (B, L, D) for unified processing
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(-1)

        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

        x = (x - self.mean) / self.stdev

        if self.affine:
            x = x * self.affine_weight + self.affine_bias

        if is_2d:
            x = x.squeeze(-1)
        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize prediction tensor back to original data scale.
        
        Args:
            x: Tensor of shape (B, H) or (B, H, D).
        """
        if self.mean is None or self.stdev is None:
            raise RuntimeError("RevIN must call normalize before denormalize.")

        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(-1)

        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + torch.where(self.affine_weight == 0, self.eps, torch.zeros_like(self.affine_weight)))

        x = x * self.stdev + self.mean

        if is_2d:
            x = x.squeeze(-1)
        return x


class ResidualBlock(nn.Module):
    """Dense Residual Block with LayerNorm, ReLU/GELU, and Dropout."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + self.skip(x)


class TiDE(nn.Module):
    """Time-series Dense Encoder (TiDE) with RevIN support.
    
    Reference:
      Das et al., "Long-term Forecasting with TiDE: Time-series Dense Encoder",
      Transactions on Machine Learning Research (TMLR), 2023.
    """

    def __init__(
        self,
        lookback_len: int = 168,
        horizon: int = 24,
        past_cov_dim: int = 0,
        future_cov_dim: int = 0,
        feature_dim: int = 16,
        hidden_dim: int = 256,
        decoder_dim: int = 128,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
        use_revin: bool = True,
        use_past_covariates: bool = True,
        use_future_covariates: bool = True,
    ):
        super().__init__()
        self.lookback_len = lookback_len
        self.horizon = horizon
        self.past_cov_dim = past_cov_dim
        self.future_cov_dim = future_cov_dim
        self.feature_dim = feature_dim
        self.use_revin = use_revin
        self.use_past_cov = use_past_covariates and (past_cov_dim > 0)
        self.use_future_cov = use_future_covariates and (future_cov_dim > 0)

        # 1. Normalization
        if self.use_revin:
            self.revin = RevIN(num_features=1, affine=True)

        # 2. Covariate Projections
        if self.use_past_cov:
            self.past_cov_proj = nn.Linear(past_cov_dim, feature_dim)
        if self.use_future_cov:
            self.future_cov_proj = nn.Linear(future_cov_dim, feature_dim)

        # 3. Dense Encoder
        encoder_input_dim = lookback_len
        if self.use_past_cov:
            encoder_input_dim += lookback_len * feature_dim
        if self.use_future_cov:
            encoder_input_dim += horizon * feature_dim

        encoder_layers = [ResidualBlock(encoder_input_dim, hidden_dim, dropout=dropout)]
        for _ in range(num_encoder_layers - 1):
            encoder_layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout=dropout))
        self.encoder = nn.Sequential(*encoder_layers)

        # 4. Dense Decoder
        self.decoder_dim = decoder_dim
        decoder_layers = [ResidualBlock(hidden_dim, decoder_dim * horizon, dropout=dropout)]
        for _ in range(num_decoder_layers - 1):
            decoder_layers.append(ResidualBlock(decoder_dim * horizon, decoder_dim * horizon, dropout=dropout))
        self.decoder = nn.Sequential(*decoder_layers)

        # 5. Temporal Projection (Per-step output head)
        step_input_dim = decoder_dim
        if self.use_future_cov:
            step_input_dim += feature_dim

        self.temporal_head = nn.Sequential(
            nn.Linear(step_input_dim, decoder_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim // 2, 1),
        )

        # 6. Direct Linear Residual Skip
        self.residual_skip = nn.Linear(lookback_len, horizon)

    def forward(
        self,
        x_past_target: torch.Tensor,
        x_past_cov: Optional[torch.Tensor] = None,
        x_future_cov: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of TiDE.
        
        Args:
            x_past_target: Tensor of shape (B, L)
            x_past_cov: Optional Tensor of shape (B, L, past_cov_dim)
            x_future_cov: Optional Tensor of shape (B, H, future_cov_dim)
        
        Returns:
            y_pred: Tensor of shape (B, H)
        """
        B = x_past_target.size(0)

        # Step 1: RevIN Normalization on historical target
        if self.use_revin:
            norm_target = self.revin.normalize(x_past_target)
        else:
            norm_target = x_past_target

        # Step 2: Feature Projections
        encoder_inputs = [norm_target]

        if self.use_past_cov and x_past_cov is not None:
            proj_past = self.past_cov_proj(x_past_cov)  # (B, L, feature_dim)
            encoder_inputs.append(proj_past.reshape(B, -1))

        if self.use_future_cov and x_future_cov is not None:
            proj_future = self.future_cov_proj(x_future_cov)  # (B, H, feature_dim)
            # If past dynamic future covariates are also available in x_past_cov, project or slice
            # Here we include future horizon projection in the encoder
            encoder_inputs.append(proj_future.reshape(B, -1))

        # Step 3: Dense Encoder -> Global Embedding
        enc_in = torch.cat(encoder_inputs, dim=-1)
        global_embed = self.encoder(enc_in)  # (B, hidden_dim)

        # Step 4: Dense Decoder -> (B, H, decoder_dim)
        dec_out = self.decoder(global_embed)  # (B, H * decoder_dim)
        dec_out = dec_out.reshape(B, self.horizon, self.decoder_dim)  # (B, H, decoder_dim)

        # Step 5: Temporal Projection Per Step
        if self.use_future_cov and x_future_cov is not None:
            step_inputs = torch.cat([dec_out, proj_future], dim=-1)  # (B, H, decoder_dim + feature_dim)
        else:
            step_inputs = dec_out

        head_out = self.temporal_head(step_inputs).squeeze(-1)  # (B, H)

        # Step 6: Direct Linear Skip + Sum
        res_out = self.residual_skip(norm_target)  # (B, H)
        y_norm_pred = head_out + res_out

        # Step 7: RevIN Denormalization back to data scale
        if self.use_revin:
            y_pred = self.revin.denormalize(y_norm_pred)
        else:
            y_pred = y_norm_pred

        return y_pred
