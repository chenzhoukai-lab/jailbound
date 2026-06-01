from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class HotFlipState:
    suffix_ids: list[int]
    suffix_text: str
    loss: float


class HotFlipSuffixOptimizer:
    """HotFlip-style token replacement helper.

    This utility is intentionally isolated from the baseline attack. It expects
    a differentiable `loss_fn(suffix_ids)` that returns `(loss, grad)` where
    `grad` is the gradient for each suffix embedding position. The caller owns
    model-specific details such as locating suffix token positions in Qwen2.5-VL.
    """

    def __init__(self, tokenizer, embedding_weight, top_k: int = 32) -> None:
        self.tokenizer = tokenizer
        self.embedding_weight = embedding_weight
        self.top_k = top_k

    def initial_ids(self, text: str, suffix_length: int) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            ids = [self.tokenizer.eos_token_id]
        ids = (ids * ((suffix_length // len(ids)) + 1))[:suffix_length]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def step(
        self,
        suffix_ids: list[int],
        loss_fn: Callable[[list[int]], tuple[float, object]],
    ) -> HotFlipState:
        import torch

        base_loss, grad = loss_fn(suffix_ids)
        emb = self.embedding_weight.detach()
        current = emb[torch.tensor(suffix_ids, device=emb.device)]
        grad = grad.to(emb.device)
        best_ids = list(suffix_ids)
        best_loss = float(base_loss)

        for pos in range(len(suffix_ids)):
            scores = torch.matmul(emb - current[pos], -grad[pos])
            top_ids = torch.topk(scores, k=min(self.top_k, scores.numel())).indices.tolist()
            for candidate_id in top_ids:
                trial = list(suffix_ids)
                trial[pos] = int(candidate_id)
                trial_loss, _ = loss_fn(trial)
                if float(trial_loss) < best_loss:
                    best_loss = float(trial_loss)
                    best_ids = trial
        return HotFlipState(best_ids, self.decode(best_ids), best_loss)

