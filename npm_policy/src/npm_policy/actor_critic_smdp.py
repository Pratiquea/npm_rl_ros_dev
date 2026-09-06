# actor_critic_pointer_pointnet_smdp.py
# ActorCriticPointerCategorical adapted for the SMDP setting.
#
# The changes from the original are as follows:
# * The observation now include the point cloud + normals and the extras.
# * Soft gather of point features using pointer logits which is differentiable
# * Force head is conditioned on soft gathered point features + embedding summary + extras
#
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def get_activation(name: str):
    name = name.lower()
    return {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }.get(name, nn.ELU())


class PointNetBlockStable(nn.Module):
    """Shared per-point MLP: [B, N, in_ch] -> [B, N, d]."""

    def __init__(self, in_ch: int, hidden: list[int] = [64, 128], out_ch: int = 128):
        super().__init__()
        dims = [in_ch] + hidden + [out_ch]
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.SiLU(),
                nn.LayerNorm(dims[i+1]),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        B, N, C = pts.shape
        x = pts.reshape(B * N, C)
        x = self.net(x)                # [B*N, d]
        return x.reshape(B, N, -1)     # [B, N, d]


def compute_pointer_logits(phi, q, temp=1.0, clamp=30.0):
    """
    phi:  [B,N,d]  per-point features
    q:    [B,d]    query
    Returns: [B, N] logits (no masking)
    """
    B, N, d = phi.shape
    # Normalize features by dot products to [-1,1], then scale like attention
    qn = F.normalize(q, p=2, dim=-1)                  # [B,d]
    phin = F.normalize(phi, p=2, dim=-1)                  # [B,N,d]
    scores = (phin * qn.unsqueeze(1)).sum(-1) * (d ** 0.5)  # [B,N]
    if temp != 1.0:
        scores = scores / temp

    scores = torch.clamp(scores, min=-clamp, max=clamp)
    # replace any accidental NaNs with zero and report via console
    if torch.isnan(scores).any():
        n_nan = torch.isnan(scores).sum().item()
        print(f"[WARNING] {n_nan} NaN values in pointer logits, replacing with zero")
    scores = torch.nan_to_num(scores, nan=0.0, posinf=clamp, neginf=-clamp)
    return scores


class ActorCriticPointerCategoricalSMDP(nn.Module):
    """
    HYBRID policy for SMDP:
      * Discrete pointer over N points (Categorical)
      * Continuous controls: [fx, fy, fz] (Gaussian, 3-dim)
      * Force head is conditioned on soft-gathered point features + embedding summary + extras

    Observation layout (flat):
        [pts (N*C), normals (N*C), extras]

    Returned action: [index, fx, fy, fz] of shape [B, 4]
    """

    is_recurrent = False
    distribution_type = "hybrid"

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,               # 4 for SMDP: 1 discrete + 3 continuous
        num_points: int,                # N = pcl_num_points
        point_dim: int,                 # C = 3
        pointnet_hidden: list[int] = [64, 128],
        embed_dim: int = 128,
        query_hidden_dims: list[int] = [128],
        critic_hidden_dims: list[int] = [256, 256],
        actor_hidden_dims: list[int] = [128, 128],
        activation: str = "relu",
        pool_mode: str = "mean+max",
        init_noise_std: float = 1.0,
        state_dependent_std: bool = False,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticPointerCategoricalSMDP.__init__ got unexpected arguments, ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()
        self.N = num_points
        self.C = point_dim
        self.num_actions = num_actions - 1  # continuous part only (fx, fy, fz = 3)
        self.pool_mode = pool_mode
        self.state_dependent_std = state_dependent_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.act_fx = get_activation(activation)
        d = embed_dim

        # Per-point encoder
        pointnet_in_ch = 2 * self.C  # points + normals
        self.pointnet = PointNetBlockStable(pointnet_in_ch, hidden=pointnet_hidden, out_ch=d)

        # Pooled context
        pooled_dim = d * 2 if pool_mode == "mean+max" else d

        # Figure out extras dimension; obs = [pc (N*C), normals (N*C), extras]
        pcl_obs_dims = self.N * self.C * 2  # pts + normals
        extras_dim_actor  = num_actor_obs  - pcl_obs_dims
        extras_dim_critic = num_critic_obs - pcl_obs_dims
        assert extras_dim_actor >= 0, (
            f"num_actor_obs ({num_actor_obs}) too small for N={self.N}, C={self.C}. "
            f"Expected at least {pcl_obs_dims} for pcl+normals."
        )
        assert extras_dim_critic >= 0, (
            f"num_critic_obs ({num_critic_obs}) is smaller than PCL dims "
            f"Expected at least {pcl_obs_dims} for pcl+normals."
        )

        # Query head
        q_in = pooled_dim + extras_dim_actor
        q_layers: list[nn.Module] = []
        last = q_in
        for h in query_hidden_dims:
            q_layers += [nn.Linear(last, h), self.act_fx]
            last = h
        q_layers += [nn.Linear(last, d)]      # query q in R^d
        self.query_net = nn.Sequential(*q_layers)

        # Gaussian continuous-head (3 outputs: fx, fy, fz)
        force_in = d + pooled_dim + extras_dim_actor

        ga_layers: list[nn.Module] = []
        last = force_in
        for h in actor_hidden_dims:
            ga_layers += [nn.Linear(last, h), self.act_fx]
            last = h

        if self.state_dependent_std:
            # output both mean and log_std
            ga_layers += [nn.Linear(last, 2 * self.num_actions)]
        else:
            ga_layers += [nn.Linear(last, self.num_actions)]
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(self.num_actions))
            )
        self.actor = nn.Sequential(*ga_layers)

        # Critic
        v_in = pooled_dim + extras_dim_critic
        critic_layers: list[nn.Module] = [nn.Linear(v_in, critic_hidden_dims[0]), self.act_fx]
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers += [nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]), self.act_fx]
        critic_layers += [nn.Linear(critic_hidden_dims[-1], 1)]
        self.critic = nn.Sequential(*critic_layers)

        # Distribution holders
        self._cat: Categorical | None = None
        self._gauss: Normal | None = None

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def _parse_obs(self, obs: torch.Tensor):
        B = obs.shape[0]
        off = 0
        # Point cloud
        pts_flat = obs[:, off: off + self.N * self.C]
        off += self.N * self.C
        pts = pts_flat.view(B, self.N, self.C)              # [B, N, C]
        # Normals
        nrm_flat = obs[:, off: off + self.N * self.C]
        off += self.N * self.C
        normals = nrm_flat.view(B, self.N, self.C)          # [B, N, C]
        # Extras, i.e. everything remaining
        extras = obs[:, off:]
        # concat pos and normal
        pts_and_normals = torch.cat([pts, normals], dim=-1)  # [B, N, 2*C]

        return pts_and_normals, extras

    def _encode(self, pts_and_normals: torch.Tensor):
        phi = self.pointnet(pts_and_normals)  # [B, N, d]
        if self.pool_mode == "mean":
            g = phi.mean(dim=1)
        elif self.pool_mode == "max":
            g = phi.max(dim=1).values
        elif self.pool_mode == "mean+max":
            g = torch.cat([phi.mean(dim=1), phi.max(dim=1).values], dim=-1)
        else:
            raise ValueError(f"Unknown pool_mode {self.pool_mode}")
        return phi, g

    def update_distribution(self, observations: torch.Tensor, **kwargs):
        pts_and_normals, extras = self._parse_obs(observations)
        phi, g = self._encode(pts_and_normals)
        h = torch.cat([g, extras], dim=-1) if extras.numel() > 0 else g

        # Pointer logits
        q = self.query_net(h)                                    # [B, d]
        scores = compute_pointer_logits(phi, q, temp=1.0, clamp=30.0)
        self._cat = Categorical(logits=scores)

        alpha = self._cat.probs                                     # [B, N]
        c = (alpha.unsqueeze(-1) * phi).sum(dim=1)                  # [B, d]

        # force head, conditioned on [c, g, extras]
        force_input = torch.cat([c, g, extras], dim=-1) \
            if extras.numel() > 0 else torch.cat([c, g], dim=-1)

        try:
            force_out = self.actor(force_input)

            if self.state_dependent_std:
                mean, raw_log_std = force_out.chunk(2, dim=-1)
                log_std = torch.clamp(raw_log_std, min=self.log_std_min, max=self.log_std_max)
                std = log_std.exp()
            else:
                mean = force_out
                std = self.log_std.exp().expand_as(mean)

            self._gauss = Normal(mean, std)

        except Exception as e:
            print("Error in actor head:", e)
            print("h:", h)
            print("h shape:", h.shape)
            print(" max h:", torch.max(h, dim=0))
            print(" min h:", torch.min(h, dim=0))
            print(" std {}".format(std))
            print(" max std:", torch.max(std, dim=0))
            print(" min std:", torch.min(std, dim=0))
            print(" mean {}".format(mean))
            print(" max mean:", torch.max(mean, dim=0))
            print(" min mean:", torch.min(mean, dim=0))
            raise e

        return self._cat, self._gauss

    def act(self, observations: torch.Tensor, **kwargs):
        cat, gauss = self.update_distribution(observations)
        idx = cat.sample()                               # [B]
        a_cont = gauss.sample()                          # [B, A-1]
        # Return single tensor: [index, cont...]; index as float for storage/env
        action = torch.cat([idx.unsqueeze(-1).float(), a_cont], dim=-1)  # [B, A]
        return action

    def act_inference(self, observations: torch.Tensor):
        cat, gauss = self.update_distribution(observations)
        idx = torch.argmax(cat.logits, dim=-1)
        return torch.cat([idx.unsqueeze(-1).float(), gauss.mean], dim=-1)
    def get_actions_log_prob(self, actions: torch.Tensor):
        if actions.dim() == 1:
            raise RuntimeError("Expected actions with shape [B, num_actions]")

        idx_f = actions[:, 0]
        a_cont = actions[:, 1:]

        idx_f = torch.nan_to_num(idx_f, nan=0.0, posinf=0.0, neginf=0.0)
        idx = torch.round(idx_f).clamp_(0, self.N - 1).long()   # [B]

        logp_cat = self._cat.log_prob(idx)   # [B]
        logp_cat = torch.nan_to_num(logp_cat, nan=-1e9, posinf=-1e9, neginf=-1e9)

        logp_gauss = self._gauss.log_prob(a_cont).sum(-1)  # [B]
        # clamp extremes to avoid +inf combining with very negative cat term
        logp_gauss = torch.clamp(
            torch.nan_to_num(logp_gauss, nan=-1e9, posinf=50.0, neginf=-50.0), -50.0, 50.0
        )

        total = logp_cat + logp_gauss
        total = torch.nan_to_num(total, nan=-1e9, posinf=-1e9, neginf=-1e9)
        return total

    def _gauss_mean_std(self):
        assert self._gauss is not None
        return self._gauss.mean, self._gauss.stddev

    def _pad_to_action_shape(self, x, pad_value):
        if getattr(self, "distribution_type", "gaussian") == "hybrid":
            B = x.size(0)
            pad = torch.full((B, 1), pad_value, device=x.device, dtype=x.dtype)
            return torch.cat([pad, x], dim=-1)  # [B, A]
        return x

    @property
    def action_mean(self):
        mean_cont, _ = self._gauss_mean_std()     # [B, A-1]
        return self._pad_to_action_shape(mean_cont, pad_value=0.0)

    @property
    def action_std(self):
        _, std_cont = self._gauss_mean_std()      # [B, A-1]
        return self._pad_to_action_shape(std_cont, pad_value=1.0)

    @property
    def entropy(self):
        assert self._cat is not None and self._gauss is not None
        return self._cat.entropy() + self._gauss.entropy().sum(-1)  # [B]

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        pts_and_normals, extras = self._parse_obs(critic_observations)
        _, g = self._encode(pts_and_normals)
        h = torch.cat([g, extras], dim=-1) if extras.numel() > 0 else g
        return self.critic(h)
