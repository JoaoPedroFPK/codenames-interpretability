"""In-repo tuned-lens translators (Belrose et al. 2023, minimal reimpl).

Per-layer affine (A_l, b_l), identity-initialised, trained to minimise
KL(final logits ‖ Unembed(FinalLN(A_l h_l + b_l))) on generic text — the
translators match the model's OWN final prediction and never see the human
labels (lens_spec §6/§9 no-leakage argument). The optimiser learns the
deviation D_l = A_l - I, so AdamW's weight decay regularises toward the
identity (a no-op translator), never toward the zero matrix. Loss values
are mean per-position KL in nats, averaged over layers; a held-out chunk
split tracks validation KL for convergence checks. Kept in-repo (~100
lines of torch) so the pinned Colab stack gains no new heavyweight
dependency.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TunedLensConfig:
    seq_len: int = 512
    n_steps: int = 1000
    seqs_per_step: int = 4
    positions_per_seq: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-3
    val_every: int = 25
    val_seqs: int = 8
    seed: int = 2026


@dataclass
class TunedLens:
    A: np.ndarray            # [L, d, d] fp32
    b: np.ndarray            # [L, d]   fp32
    history: List[float] = field(default_factory=list)
    val_history: List[float] = field(default_factory=list)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path, A=self.A, b=self.b,
            history=np.asarray(self.history, dtype=np.float64),
            val_history=np.asarray(self.val_history, dtype=np.float64))

    @classmethod
    def load(cls, path: str) -> "TunedLens":
        z = np.load(path)
        return cls(A=z["A"].astype(np.float32), b=z["b"].astype(np.float32),
                   history=list(z["history"]),
                   val_history=list(z["val_history"])
                   if "val_history" in z.files else [])

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

    # A_l = I + D_l: learning the deviation makes AdamW's decay-toward-zero
    # act as decay-toward-identity on the translator.
    D = torch.zeros(L, d, d, device=device, requires_grad=True)
    b = torch.zeros(L, d, device=device, requires_grad=True)
    opt = torch.optim.AdamW([D, b], lr=config.lr,
                            weight_decay=config.weight_decay)

    chunks = _chunk_token_ids(tokenizer, texts, config.seq_len, config.seed)
    if len(chunks) == 0:
        raise ValueError("Training texts produced zero full-length chunks.")
    n_val = min(config.val_seqs, len(chunks) // 10)
    val_chunks, chunks = chunks[:n_val], chunks[n_val:]
    if len(chunks) == 0:
        raise ValueError("Training texts produced no chunks after the "
                         "validation split.")
    rng = np.random.default_rng(config.seed)
    history: List[float] = []
    val_history: List[float] = []

    def _mean_kl(input_ids, pos):
        """Mean per-position KL(final ‖ lens), averaged over layers.

        Validation only: called under the caller's ``torch.no_grad()``, so no
        graph is retained regardless of the accumulate-then-return shape.
        Training uses ``_backward_and_total`` instead (see its docstring).
        """
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True,
                        return_dict=True)
        teacher = F.log_softmax(
            out.logits[:, pos, :].float(), dim=-1).detach()      # [B, P, V]
        teacher = teacher.reshape(-1, teacher.shape[-1])         # [B*P, V]
        loss = torch.zeros((), device=device)
        for layer in range(L):
            h = out.hidden_states[layer][:, pos, :].float().detach()
            translated = h + h @ D[layer].T + b[layer]
            student = F.log_softmax(
                lm_head(norm(translated.to(norm.weight.dtype))).float(),
                dim=-1).reshape(-1, teacher.shape[-1])
            loss = loss + F.kl_div(student, teacher, log_target=True,
                                   reduction="batchmean")
        return loss / L

    def _backward_and_total(input_ids, pos) -> float:
        """One optimiser step's backward, one layer's graph at a time.

        Summing all L layers' KL terms into one graph before a single
        ``backward()`` (the original design) holds L simultaneous copies of
        the vocab-sized logits (``[B*P, V]``) until that one call returns --
        for a large vocabulary and many layers (Llama: d=4096, L=32,
        V=128256) that exceeded 39 GB on an A100 mid-training. Each term
        depends only on its OWN leaf parameters (``D[layer]``, ``b[layer]``)
        and a hidden state already detached from the frozen forward pass, so
        the terms share no graph nodes; calling ``backward()`` once per layer
        accumulates into ``D.grad``/``b.grad`` exactly what one combined
        backward would (proved in ``test_per_layer_backward_matches_combined
        _backward_gradient``), while never holding more than one layer's
        logits in memory.
        """
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True,
                        return_dict=True)
        teacher = F.log_softmax(
            out.logits[:, pos, :].float(), dim=-1).detach()
        teacher = teacher.reshape(-1, teacher.shape[-1])
        total = 0.0
        for layer in range(L):
            h = out.hidden_states[layer][:, pos, :].float().detach()
            translated = h + h @ D[layer].T + b[layer]
            student = F.log_softmax(
                lm_head(norm(translated.to(norm.weight.dtype))).float(),
                dim=-1).reshape(-1, teacher.shape[-1])
            layer_loss = F.kl_div(student, teacher, log_target=True,
                                  reduction="batchmean") / L
            layer_loss.backward()
            total += float(layer_loss.detach())
        return total

    for step in range(config.n_steps):
        batch = chunks[rng.integers(0, len(chunks),
                                    size=config.seqs_per_step)]
        input_ids = torch.tensor(batch, dtype=torch.long, device=device)
        pos = rng.integers(1, config.seq_len,
                           size=min(config.positions_per_seq, config.seq_len - 1))
        opt.zero_grad()
        loss_value = _backward_and_total(input_ids, pos)
        opt.step()
        history.append(loss_value)

        last = step + 1 == config.n_steps
        if n_val and (last or (step + 1) % config.val_every == 0):
            val_ids = torch.tensor(val_chunks, dtype=torch.long,
                                   device=device)
            val_pos = np.arange(1, config.seq_len,
                                max(1, (config.seq_len - 1)
                                    // config.positions_per_seq))
            with torch.no_grad():
                val_history.append(float(_mean_kl(val_ids, val_pos).cpu()))
        if last or (step + 1) % 25 == 0:
            val_msg = (f" val={val_history[-1]:.4f}" if val_history else "")
            print(f"  tuned-lens step {step + 1}/{config.n_steps} "
                  f"loss={history[-1]:.4f}{val_msg}")

    eye = np.eye(d, dtype=np.float32)[None, :, :]
    return TunedLens(A=(D.detach().cpu().numpy() + eye).astype(np.float32),
                     b=b.detach().cpu().numpy().astype(np.float32),
                     history=history, val_history=val_history)
