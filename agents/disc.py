import torch

from agents.base import BaseAgent
from agents.dueling_q import DuelingQDecoder


class Disc(BaseAgent):
    def __init__(
        self,
        hidden_dim=128,
        n_heads=8,
        ff_dim=256,
        dropout=0.0,
        logit_scale=5.0,
        value_uses_embedding=True,
        epsilon=0.05,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        self.epsilon = float(epsilon)

        self.q_decoder = DuelingQDecoder(
            hidden_dim=hidden_dim,
            aux_dim=1,
            dropout=dropout,
            logit_scale=logit_scale,
            value_uses_embedding=value_uses_embedding,
        )

    def _prepare_progress_for_candidates(self, h_surr, progress):
        h_surr = self._ensure_batch(h_surr)

        progress_state = self._prepare_progress(
            progress=progress,
            device=h_surr.device,
            dtype=h_surr.dtype,
            batch_size=h_surr.size(0),
        )

        return progress_state.unsqueeze(1).expand(-1, h_surr.size(1), -1)

    def _actor_logits(self, h_surr, progress, candidate_mask=None):
        h_surr = self._ensure_batch(h_surr)

        progress_state = self._prepare_progress(
            progress=progress,
            device=h_surr.device,
            dtype=h_surr.dtype,
            batch_size=h_surr.size(0),
        )

        progress_cand = progress_state.unsqueeze(1).expand(-1, h_surr.size(1), -1)

        q_values = self.q_decoder(
            h_cand=h_surr,
            aux_cand=progress_cand,
            aux_state=progress_state,
            candidate_mask=candidate_mask,
        )

        return q_values

    def _epsilon_greedy_ranking(self, q_values, epsilon=0.05, candidate_mask=None):
        """
        Return ranking [B, N] by epsilon-greedy.

        With probability epsilon:
            random valid ranking
        Otherwise:
            descending Q ranking
        """
        bsz, n_cand = q_values.shape
        device = q_values.device

        if candidate_mask is not None:
            q_values = q_values.masked_fill(~candidate_mask, -1e9)

        greedy_ranking = torch.argsort(q_values, dim=-1, descending=True)

        random_rankings = []
        for b in range(bsz):
            if candidate_mask is None:
                perm = torch.randperm(n_cand, device=device)
            else:
                valid_idx = torch.nonzero(candidate_mask[b], as_tuple=False).squeeze(-1)
                invalid_idx = torch.nonzero(~candidate_mask[b], as_tuple=False).squeeze(-1)

                valid_perm = valid_idx[torch.randperm(valid_idx.numel(), device=device)]
                perm = torch.cat([valid_perm, invalid_idx], dim=0)

            random_rankings.append(perm)

        random_ranking = torch.stack(random_rankings, dim=0)

        use_random = torch.rand(bsz, device=device) < float(epsilon)

        ranking = torch.where(
            use_random.unsqueeze(1),
            random_ranking,
            greedy_ranking,
        )

        return ranking

    def decode_ranking(
        self,
        h_surr,
        progress,
        target_ranking=None,
        decode_type="epsilon_greedy",
        max_decode_steps=None,
        candidate_mask=None,
        epsilon=None,
    ):
        h_surr = self._ensure_batch(h_surr)

        q_values = self._actor_logits(
            h_surr=h_surr,
            progress=progress,
            candidate_mask=candidate_mask,
        )

        if epsilon is None:
            epsilon = self.epsilon

        if target_ranking is not None:
            ranking = target_ranking

        elif decode_type == "epsilon_greedy":
            ranking = self._epsilon_greedy_ranking(
                q_values=q_values,
                epsilon=epsilon,
                candidate_mask=candidate_mask,
            )

        elif decode_type in ["q_greedy", "greedy"]:
            ranking = torch.argsort(q_values, dim=-1, descending=True)

        elif decode_type == "softmax_sample":
            decode_steps = q_values.size(1) if max_decode_steps is None else min(
                int(max_decode_steps), q_values.size(1)
            )
            ranking = self._sample_actions_without_replacement(q_values, decode_steps)

        else:
            raise ValueError(f"Unknown decode_type: {decode_type}")

        if max_decode_steps is not None:
            ranking = ranking[:, : min(int(max_decode_steps), ranking.size(1))]

        return {
            "logits": q_values,
            "q_values": q_values,
            "ranking": ranking,
        }

    def _sample_actions_without_replacement(self, logits, k):
        bsz, n_sur = logits.shape
        selected_mask = torch.zeros(bsz, n_sur, dtype=torch.bool, device=logits.device)
        actions = []
        decode_steps = min(int(k), n_sur)

        for _ in range(decode_steps):
            masked_logits = logits.masked_fill(selected_mask, -1e9)
            dist = torch.distributions.Categorical(logits=masked_logits)
            chosen_idx = dist.sample()
            actions.append(chosen_idx.unsqueeze(1))
            selected_mask = selected_mask.scatter(1, chosen_idx.unsqueeze(1), True)

        return torch.cat(actions, dim=1)

    def forward(
        self,
        x_true,
        y_true,
        x_sur,
        y_sur,
        sigma_sur,
        progress,
        lower_bound,
        upper_bound,
        target_ranking=None,
        decode_type="epsilon_greedy",
        max_decode_steps=None,
        candidate_mask=None,
        epsilon=None,
    ):
        encoded = self.encode(
            x_true=x_true,
            y_true=y_true,
            x_sur=x_sur,
            y_sur=y_sur,
            sigma_sur=sigma_sur,
            progress=progress,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        decoded = self.decode_ranking(
            h_surr=encoded["H_surr"],
            progress=encoded["progress"],
            target_ranking=target_ranking,
            decode_type=decode_type,
            max_decode_steps=max_decode_steps,
            candidate_mask=candidate_mask,
            epsilon=epsilon,
        )

        encoded.update(decoded)
        return encoded