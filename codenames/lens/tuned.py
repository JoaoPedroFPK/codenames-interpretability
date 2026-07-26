"""In-repo tuned-lens translators (Belrose et al. 2023, minimal reimpl).

Per-layer affine (A_l, b_l), identity-initialised, trained to minimise
KL(final logits ‖ Unembed(FinalLN(A_l h_l + b_l))) on generic text — the
translators match the model's OWN final prediction and never see the human
labels (lens_spec §6/§9 no-leakage argument). Kept in-repo (~100 lines of
torch) so the pinned Colab stack gains no new heavyweight dependency.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TunedLensConfig:
    seq_len: int = 512
    n_steps: int = 250
    seqs_per_step: int = 4
    positions_per_seq: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-3
    seed: int = 2026


@dataclass
class TunedLens:
    A: np.ndarray            # [L, d, d] fp32
    b: np.ndarray            # [L, d]   fp32
    history: List[float] = field(default_factory=list)

    def save(self, path: str) -> None:
        np.savez_compressed(path, A=self.A, b=self.b,
                            history=np.asarray(self.history, dtype=np.float64))

    @classmethod
    def load(cls, path: str) -> "TunedLens":
        z = np.load(path)
        return cls(A=z["A"].astype(np.float32), b=z["b"].astype(np.float32),
                   history=list(z["history"]))

    def translate(self, H: np.ndarray, layer: int) -> np.ndarray:
        if layer >= self.A.shape[0]:      # final hidden state: identity
            return H.astype(np.float32, copy=False)
        return H.astype(np.float32) @ self.A[layer].T + self.b[layer]


def _chunk_token_ids(tokenizer, texts, seq_len, seed):
    ids = []
    for t in texts:
        ids.extend(tokenizer.encode(t, add_special_tokens=False))
    n_chunks = len(ids) // seq_len
    chunks = np.array(ids[: n_chunks * seq_len]).reshape(n_chunks, seq_len)
    rng = np.random.default_rng(seed)
    rng.shuffle(chunks)
    return chunks


def train_tuned_lens(model, tokenizer, texts, config: TunedLensConfig,
                     device: Optional[str] = None) -> TunedLens:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config.seed)
    L = model.config.num_hidden_layers
    d = model.config.hidden_size
    norm = model.base_model.norm
    lm_head = model.get_output_embeddings()

    A = torch.stack([torch.eye(d) for _ in range(L)]).to(device) \
        .requires_grad_(True)
    b = torch.zeros(L, d, device=device, requires_grad=True)
    opt = torch.optim.AdamW([A, b], lr=config.lr,
                            weight_decay=config.weight_decay)

    chunks = _chunk_token_ids(tokenizer, texts, config.seq_len, config.seed)
    if len(chunks) == 0:
        raise ValueError("Training texts produced zero full-length chunks.")
    rng = np.random.default_rng(config.seed)
    history: List[float] = []

    for step in range(config.n_steps):
        batch = chunks[rng.integers(0, len(chunks),
                                    size=config.seqs_per_step)]
        input_ids = torch.tensor(batch, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True,
                        return_dict=True)
        pos = rng.integers(1, config.seq_len,
                           size=min(config.positions_per_seq, config.seq_len - 1))
        teacher = F.log_softmax(
            out.logits[:, pos, :].float(), dim=-1).detach()      # [B, P, V]

        loss = torch.zeros((), device=device)
        for layer in range(L):
            h = out.hidden_states[layer][:, pos, :].float().detach()
            student = F.log_softmax(
                lm_head(norm((h @ A[layer].T + b[layer])
                             .to(norm.weight.dtype))).float(), dim=-1)
            loss = loss + F.kl_div(student, teacher, log_target=True,
                                   reduction="batchmean")
        loss = loss / L
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.detach().cpu()))
        if (step + 1) % 25 == 0:
            print(f"  tuned-lens step {step + 1}/{config.n_steps} "
                  f"loss={history[-1]:.4f}")

    return TunedLens(A=A.detach().cpu().numpy().astype(np.float32),
                     b=b.detach().cpu().numpy().astype(np.float32),
                     history=history)
