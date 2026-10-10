"""World-State Predictor Module for NeuroGraph-LLM."""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurograph.config import WorldModelConfig


class WorldStatePredictor(nn.Module):
    """Predicts next environment state from current state and candidate action,

    with epistemic and aleatoric uncertainty estimation.
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.hidden_dim = config.hidden_dim
        self.predict_uncertainty = config.predict_uncertainty

        # State and Action Encoders
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.action_encoder = nn.Embedding(self.action_dim, self.hidden_dim)

        # Transition dynamics network
        self.transition_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        # Output prediction heads
        self.mean_head = nn.Linear(self.hidden_dim, self.state_dim)
        if self.predict_uncertainty:
            self.logvar_head = nn.Linear(self.hidden_dim, self.state_dim)

    def forward(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next state distribution.

        Args:
            current_state (torch.Tensor): [B, state_dim]
            action (torch.Tensor): [B] integer action indices in [0, action_dim - 1]

        Returns:
            predicted_mean (torch.Tensor): [B, state_dim]
            predicted_uncertainty (torch.Tensor): [B, state_dim] standard deviation estimate.
        """
        assert current_state.shape[-1] == self.state_dim, (
            f"Expected state dim {self.state_dim}, got {current_state.shape[-1]}"
        )

        s_emb = self.state_encoder(current_state)  # [B, hidden_dim]
        a_emb = self.action_encoder(action)        # [B, hidden_dim]

        joint = torch.cat([s_emb, a_emb], dim=-1) # [B, hidden_dim * 2]
        latent_next = self.transition_net(joint)   # [B, hidden_dim]

        # Residual next-state prediction: predict delta + current_state
        delta = self.mean_head(latent_next)
        pred_mean = current_state + delta

        if self.predict_uncertainty:
            logvar = self.logvar_head(latent_next)
            # Bound log-variance for numerical stability
            logvar = torch.clamp(logvar, min=-10.0, max=5.0)
            uncertainty = torch.exp(0.5 * logvar)
            return pred_mean, uncertainty
        else:
            uncertainty = torch.zeros_like(pred_mean)
            return pred_mean, uncertainty

    def compute_loss(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor,
        target_next_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute transition prediction loss.

        Uses Gaussian Negative Log-Likelihood if uncertainty prediction is enabled,
        or standard MSE loss otherwise.
        """
        s_emb = self.state_encoder(current_state)
        a_emb = self.action_encoder(action)
        joint = torch.cat([s_emb, a_emb], dim=-1)
        latent_next = self.transition_net(joint)
        delta = self.mean_head(latent_next)
        pred_mean = current_state + delta

        if self.predict_uncertainty:
            logvar = self.logvar_head(latent_next)
            logvar = torch.clamp(logvar, min=-10.0, max=5.0)
            inv_var = torch.exp(-logvar)
            # Gaussian NLL: 0.5 * (inv_var * (target - mean)^2 + logvar)
            sq_err = (target_next_state - pred_mean) ** 2
            nll_loss = 0.5 * torch.mean(inv_var * sq_err + logvar)
            mse_loss = torch.mean(sq_err)
            uncertainty = torch.exp(0.5 * logvar)
            return {
                "loss": nll_loss,
                "mse": mse_loss,
                "pred_mean": pred_mean,
                "uncertainty": uncertainty,
            }
        else:
            mse_loss = F.mse_loss(pred_mean, target_next_state)
            return {
                "loss": mse_loss,
                "mse": mse_loss,
                "pred_mean": pred_mean,
                "uncertainty": torch.zeros_like(pred_mean),
            }

    @torch.no_grad()
    def evaluate_transitions(
        self,
        current_states: torch.Tensor,
        actions: torch.Tensor,
        target_next_states: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate world model accuracy on held-out transitions."""
        self.eval()
        outputs = self.compute_loss(current_states, actions, target_next_states)
        mse = float(outputs["mse"].item())
        avg_uncertainty = float(outputs["uncertainty"].mean().item())
        return {
            "eval_mse": mse,
            "avg_uncertainty": avg_uncertainty,
        }
