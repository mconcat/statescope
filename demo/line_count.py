"""Line counting in Qwen3.5-0.8B, followed through the released dictionaries: L8 MLP -> L9 GDN head 10 -> L9 MLP.

Task. A text is wrapped to a fixed width. At the last word of a line the model has to predict a line break, so it must know how many
characters the line already holds. Each item pairs 256-token texts ending on the same words; the clean text's last line is full,
at most 6 characters short of the width, so a line break is due. In the corrupt text a line break is not due.
In the count cells ("count:60", "count:100"), the corrupt text's last line starts at a later word chosen so that it holds between
35% and 65% of the width at the last word; preceding lines are wrapped backward from that word at the same width, moving their
line breaks too. In the width cell ("width:100"), the last line starts at the same word as in the clean text, and preceding lines
are wrapped backward at 1.75 times the width.

Score. m = logsumexp(logits of newline tokens) - logsumexp(logits of tokens that start with a space), at the last token.
A patch takes one tensor of the clean run and replaces it with a value built from the corrupt run (moved onto the clean run's token
positions). share = mean(m_clean - m_patched) / mean(m_clean - m_corrupt): the part of the clean/corrupt difference that the
patched tensor carries. 0 = the patch changes nothing, 1 = the patch alone turns the clean run into the corrupt one.

The chain (found by the original project; this script takes its choice of transform, group and head as given and re-measures):
  1. L8 MLP dictionary, transform 189 fires on the first token of a line. Its write lies mostly inside the 4-dimensional frame of
     state group 11586 of resid_post.8 (the "first token of a line" group).
  2. The model's own L9 mixer reads that: swapping only transform 189's write, or only group 11586, in what the L9 mixer reads moves
     the answer about as much as swapping the whole L8 MLP output. Inside the L9 mixer, GDN head 10 carries the count.
  3. L9 MLP dictionary: transform 410 (on while the line is short) and transform 1749 (on once the line is long) turn the count into
     the line-break decision. What the L9 transforms read of the mixer's write lies mostly inside the 4-dimensional frame of
     state group 4885 of resid_mid.9: that part alone moves them about as much as the whole write, the rest moves them little
     (the rest still reaches the answer by other routes). This is measured for the original's six L9 transforms together
     (410, 1749, 6, 3720, 1394, 2275) and for 410 and 1749 alone.

    python demo/line_count.py --release <folder with state/ and mlp/> [--device cuda:0] [--out result_folder]

Needs transformers with Qwen3.5 support; on a GPU also flash-linear-attention (without it the GDN layers run a slow reference kernel).

The file has three parts after the constants: the model with hooks, the L9 mixer (a Gated DeltaNet layer) by hand, the measurements.
(The pairing helper that both the mixer part and the measurements use sits between the first two.)
"""
from __future__ import annotations

import argparse
import json
import os
import types
from dataclasses import dataclass, replace
from typing import Callable

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from statescope.config import MODELS
from statescope.lm import LanguageModel, load_language_model
from statescope.reader import PublicMolt, StateVocab

# ---------------------------------------------------------------------------------------------- constants
CELLS = ("count:60", "count:100", "width:100")
COUNT_CELLS = ("count:60", "count:100")
BATCH = 32
# HEAD, the transforms and the groups below are the original project's choices, not options: output keys such as 'l8/t189', 'l8/g11586',
# 'g4885' and 'h10o' spell these numbers out, and tests/test_demo.py maps the keys to the original's records.
HEAD = 10                          # GDN head of layer 9 that carries the count
L8_TRANSFORM, L8_GROUP = 189, 11586
L9_TRANSFORMS = (410, 1749)
L9_GROUP = 4885                    # state group of resid_mid.9 whose frame carries the count from the mixer to the L9 transforms
LINK_ROUTES = {"F2": (410, 1749, 6, 3720, 1394, 2275), "T2": L9_TRANSFORMS}   # the original's six L9 transforms; the two named above
FRAME_DIM = 4                      # dimensions of a state group's frame; the random frames of the comparison get the same
N_RANDOM_FRAMES = 5
CHAR_BIN = 10                      # width, in characters, of the bins of by_chars_in_line
MAX_CHARS = 100                    # by_chars_in_line covers lines of up to this many characters
ITEMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "line_count_items.safetensors")

Patch = Callable[[torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------------------------- the model with hooks; the score
@dataclass
class Run:
    """One forward pass over a batch (B, T). Residuals and outputs are fp32 copies of the model's bf16 tensors, raw units."""
    hidden: torch.Tensor      # (B, T, d) output of the final norm, model dtype
    resid_in8: torch.Tensor   # input of layer 8
    mix8: torch.Tensor        # output of layer 8's mixer
    mlp8: torch.Tensor        # output of layer 8's MLP
    post8: torch.Tensor       # resid_post.8 = input of layer 9
    mixer_in9: torch.Tensor   # what layer 9's mixer receives (after the layer's input norm), model dtype
    mix9: torch.Tensor        # output of layer 9's mixer
    mlp9: torch.Tensor        # output of layer 9's MLP

    @property
    def mid8(self) -> torch.Tensor:
        return self.resid_in8 + self.mix8

    @property
    def mid9(self) -> torch.Tensor:
        return self.post8 + self.mix9


def _first(out: torch.Tensor | tuple | list) -> torch.Tensor:
    """the tensor of a module's output: a module returns either the tensor or a tuple that starts with it"""
    return out[0] if isinstance(out, (tuple, list)) else out


def _keep(got: dict[str, torch.Tensor], name: str) -> Callable:
    """forward hook: store the module's output as got[name]; the output is not changed"""
    def hook(module, args, out) -> None:
        got[name] = _first(out)
    return hook


def _keep_input(got: dict[str, torch.Tensor], name: str) -> Callable:
    """forward pre-hook (with keyword arguments): store the module's input as got[name]; the input is not changed"""
    def hook(module, args, kwargs) -> None:
        got[name] = args[0] if args else kwargs["hidden_states"]
    return hook


def _replace(got: dict[str, torch.Tensor], name: str, patch: Patch | None) -> Callable:
    """forward hook: store the module's own output as got[name]; with a patch, the module then returns patch(output) instead"""
    def hook(module, args, out):
        t = _first(out)
        got[name] = t
        if patch is None:
            return None
        new = patch(t).to(t.dtype)                      # a patched tensor is rounded to the model's dtype, like the original
        return (new, *out[1:]) if isinstance(out, (tuple, list)) else new
    return hook


class HookedModel:
    """The language model with the handles the demonstration needs: its layers, the final norm and the mixer of layer 9.

    run() is one forward pass over a batch. It keeps the tensors listed in `Run` (taken at layers 8 and 9 and after the final norm)
    and can replace the output of layer 9's mixer or of layer 9's MLP on the way: a patch receives the module's own output and returns
    the tensor to use instead; everything after that module is then computed from the replacement."""

    def __init__(self, lm: LanguageModel) -> None:
        self.lm = lm
        self.model, self.layers = lm.model, lm.layers
        self.final_norm = lm.model.model.norm
        self.mixer9 = self.layers[9].linear_attn
        self.dtype = lm.model.dtype                     # bf16: the residual and every module's output are in this dtype

    @torch.inference_mode()
    def run(self, ids: torch.Tensor, patch_mix9: Patch | None = None, patch_mlp9: Patch | None = None) -> Run:
        got: dict[str, torch.Tensor] = {}
        l8, l9 = self.layers[8], self.layers[9]
        hooks = [l8.register_forward_pre_hook(_keep_input(got, "resid_in8"), with_kwargs=True),
                 l8.linear_attn.register_forward_hook(_keep(got, "mix8")), l8.mlp.register_forward_hook(_keep(got, "mlp8")),
                 l9.register_forward_pre_hook(_keep_input(got, "post8"), with_kwargs=True),
                 l9.linear_attn.register_forward_pre_hook(_keep_input(got, "mixer_in9"), with_kwargs=True),
                 l9.linear_attn.register_forward_hook(_replace(got, "mix9", patch_mix9)),
                 l9.mlp.register_forward_hook(_replace(got, "mlp9", patch_mlp9)),
                 self.final_norm.register_forward_hook(_keep(got, "hidden"))]
        try:
            self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
        finally:
            for h in hooks:
                h.remove()
        return Run(got["hidden"], got["resid_in8"].float(), got["mix8"].float(), got["mlp8"].float(), got["post8"].float(),
                   got["mixer_in9"], got["mix9"].float(), got["mlp9"].float())


class LineBreakScore:
    """The score of the task, read from the final hidden state at the last token:
    m = logsumexp(logits of line-break tokens) - logsumexp(logits of tokens that start with a space)."""

    def __init__(self, lm: LanguageModel) -> None:
        unembed = lm.model.get_output_embeddings().weight
        text = [lm.tokenizer.decode([i]) for i in range(len(lm.tokenizer))]
        self.token_text = text                                             # the text of every token of the vocabulary
        self.newline = {i for i, s in enumerate(text) if ("\n" in s or "\r" in s) and s.strip() == ""}   # line-break tokens
        space = [i for i, s in enumerate(text) if s.startswith(" ") and "\n" not in s and "\r" not in s]
        self.W_newline, self.W_space = unembed[sorted(self.newline)].float(), unembed[space].float()        # their unembedding rows

    def margin(self, hidden: torch.Tensor) -> torch.Tensor:
        """(B, T, d) final hidden states -> (B,) margin at the last token"""
        h = hidden[:, -1].float()
        return (h @ self.W_newline.T).logsumexp(1) - (h @ self.W_space.T).logsumexp(1)


# ---------------------------------------------------------------------------------------------- pairing: the corrupt run's values on the clean run's positions
def onto_clean(corrupt: torch.Tensor, clean: torch.Tensor, pairing: torch.Tensor) -> torch.Tensor:
    """corrupt, clean (B, T, ...): the corrupt run's values moved onto the clean run's positions. pairing (B, T) gives for each clean
    position the corrupt position that holds the same word piece, or -1 (then the clean value stays)."""
    shape = [1] * corrupt.dim()
    shape[0], shape[1] = corrupt.shape[0], corrupt.shape[1]
    index = pairing.clamp_min(0).view(*shape).expand_as(corrupt)
    return torch.where((pairing >= 0).view(*shape), corrupt.gather(1, index), clean)


# ---------------------------------------------------------------------------------------------- the L9 mixer (a Gated DeltaNet layer) by hand
def _qwen_module() -> types.ModuleType:
    """transformers' Qwen3.5 module: its causal convolution, its delta-rule kernel and its l2norm are used below.
    Imported on first use, so that --help works without transformers."""
    from transformers.models.qwen3_5 import modeling_qwen3_5
    return modeling_qwen3_5


def gdn_inputs(mixer: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """The mixer's own projections of its input x (B, T, d): per head q, k (B, T, H, dk), v, z (B, T, H, dv), g, beta (B, T, H)."""
    MQ = _qwen_module()
    B, T, _ = x.shape
    mixed = mixer.in_proj_qkv(x).transpose(1, 2)
    z = mixer.in_proj_z(x).reshape(B, T, -1, mixer.head_v_dim)
    b, a = mixer.in_proj_b(x), mixer.in_proj_a(x)
    mixed = MQ.causal_conv1d_fn(mixed, mixer.conv1d.weight.squeeze(1), mixer.conv1d.bias, activation=mixer.activation).transpose(1, 2)
    q, k, v = torch.split(mixed, [mixer.key_dim, mixer.key_dim, mixer.value_dim], dim=-1)
    q, k = q.reshape(B, T, -1, mixer.head_k_dim), k.reshape(B, T, -1, mixer.head_k_dim)
    v = v.reshape(B, T, -1, mixer.head_v_dim)
    beta = b.sigmoid()                                                     # how much of the new value is written
    g = -mixer.A_log.float().exp() * F.softplus(a.float() + mixer.dt_bias)  # log of the state's decay
    repeat = mixer.num_v_heads // mixer.num_k_heads
    if repeat > 1:
        q, k = q.repeat_interleave(repeat, dim=2), k.repeat_interleave(repeat, dim=2)
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta, "z": z}


def gdn_output(mixer: torch.nn.Module, p: dict[str, torch.Tensor]) -> torch.Tensor:
    """The mixer's output from its projections: the model's own delta-rule kernel, gated norm and output projection."""
    MQ = _qwen_module()
    B, T = p["q"].shape[:2]
    core, _ = MQ.torch_chunk_gated_delta_rule(p["q"], p["k"], p["v"], g=p["g"], beta=p["beta"], initial_state=None,
                                              output_final_state=False, use_qk_l2norm_in_kernel=True)
    core = mixer.norm(core.reshape(-1, mixer.head_v_dim), p["z"].reshape(-1, mixer.head_v_dim)).reshape(B, T, -1)
    return mixer.out_proj(core)


def head_state_read(p: dict[str, torch.Tensor], head: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One head's recurrence in fp32, token by token: the state S (dk, dv) decays, the delta rule overwrites what key k_t holds
    with v_t, and the query q_t reads it. Returns the read o (B, T, dv) before the head's norm, and the head's norm gate z."""
    MQ = _qwen_module()
    q = MQ.l2norm(p["q"][:, :, head].float(), dim=-1, eps=1e-6)
    q = q / (q.shape[-1] ** 0.5)
    k = MQ.l2norm(p["k"][:, :, head].float(), dim=-1, eps=1e-6)
    v, beta, g = p["v"][:, :, head].float(), p["beta"][:, :, head].float(), p["g"][:, :, head].float()
    B, T, dk = k.shape
    S = torch.zeros(B, dk, v.shape[-1], device=k.device, dtype=torch.float32)
    o = torch.zeros_like(v)
    for t in range(T):
        S = S * g[:, t].exp()[:, None, None]
        held = (S * k[:, t, :, None]).sum(1)
        S = S + k[:, t, :, None] * ((v[:, t] - held) * beta[:, t, None])[:, None, :]
        o[:, t] = (S * q[:, t, :, None]).sum(1)
    return o, p["z"][:, :, head].float()


def head_normed(mixer: torch.nn.Module, o: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """One head's read o (B, T, dv) through the mixer's gated norm with gate z, computed in the model's dtype; fp32 result."""
    B, T, dv = o.shape
    dtype = mixer.out_proj.weight.dtype
    return mixer.norm(o.reshape(-1, dv).to(dtype), z.reshape(-1, dv).to(dtype)).reshape(B, T, dv).float()


def head_swapped_write(mixer: torch.nn.Module, head: int, mixer_in_clean: torch.Tensor, mixer_in_corrupt: torch.Tensor,
                       pairing: torch.Tensor) -> torch.Tensor:
    """How the mixer's output (B, T, d) changes when one head alone reads the corrupt text. The head's read is computed by hand for
    both texts, the corrupt read is moved onto the clean positions, and both go through the head's norm with the clean gate and
    through the head's columns of the output projection. Returns swapped minus clean, fp32."""
    head_columns = mixer.out_proj.weight[:, head * mixer.head_v_dim:(head + 1) * mixer.head_v_dim].float()
    o_c, z_c = head_state_read(gdn_inputs(mixer, mixer_in_clean), head)
    o_x, _ = head_state_read(gdn_inputs(mixer, mixer_in_corrupt), head)
    return (head_normed(mixer, onto_clean(o_x, o_c, pairing), z_c) - head_normed(mixer, o_c, z_c)) @ head_columns.T


def mixer9_of(model: HookedModel, post8: torch.Tensor) -> torch.Tensor:
    """The output (B, T, d), fp32, of the model's own L9 mixer for a given resid_post.8 (B, T, d): layer 9's input norm in the
    model's dtype, then the mixer by hand."""
    seen = model.layers[9].input_layernorm(post8.to(model.dtype))
    return gdn_output(model.mixer9, gdn_inputs(model.mixer9, seen)).float()


# ---------------------------------------------------------------------------------------------- the measurements: shares, labels
def share(m_clean: torch.Tensor, m_patched: torch.Tensor, m_corrupt: torch.Tensor, n_boot: int = 1000) -> list[float]:
    """[share, 2.5 %, 97.5 %] with a bootstrap over items (the original's seed and draw)"""
    num, den = m_clean - m_patched, m_clean - m_corrupt
    draw = torch.randint(0, len(num), (n_boot, len(num)), generator=torch.Generator().manual_seed(0))
    ratio = num[draw].mean(1) / den[draw].mean(1)
    return [float(num.mean() / den.mean()), float(ratio.quantile(0.025)), float(ratio.quantile(0.975))]


def token_labels(ids: torch.Tensor, score: LineBreakScore) -> dict[str, torch.Tensor]:
    """Per token of ids (B, T): chars = characters since the last line break, this token included; newline = is a line-break token;
    line_first = follows a line-break token; known = a line break has been seen (before that, chars counts from the text's start)."""
    out: dict[str, list] = {k: [] for k in ("chars", "newline", "line_first", "known")}
    for row in ids.tolist():
        chars, seen, after_newline = 0, False, False
        r: dict[str, list] = {k: [] for k in out}
        for token in row:
            s = score.token_text[token]
            if "\n" in s:
                chars, seen = len(s) - s.rfind("\n") - 1, True
            else:
                chars += len(s)
            is_newline = token in score.newline
            r["chars"].append(chars), r["newline"].append(is_newline), r["line_first"].append(after_newline), r["known"].append(seen)
            after_newline = is_newline
        for k in out:
            out[k].append(r[k])
    return {k: torch.tensor(v) for k, v in out.items()}


def group_contribution(vocab: StateVocab, x: torch.Tensor, site: str, group: int) -> torch.Tensor:
    """what state group `group` adds to the raw residual at each token of x (B, T, d): zero where the group is not among the k active"""
    flat = x.reshape(-1, x.shape[-1])
    groups, codes = vocab.encode(flat, site)
    code = (codes * (groups == group)[:, :, None]).sum(1)
    return vocab.group_write(group, code, site, rms=vocab.token_rms(flat)).view_as(x)


def in_frame(frame: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """the part of v (..., d) inside the subspace spanned by the orthonormal rows of frame (b, d)"""
    return (v @ frame.T) @ frame


# ---------------------------------------------------------------------------------------------- the measurements: the arms of one batch
# An arm is one patched run of the clean texts: one tensor of the clean run is replaced by a value built from the corrupt run, and the
# margin is read. The functions below are called by measure() once per batch, in this order, under its torch.inference_mode().
# Suffixes _c and _x mean the clean and the corrupt run; they are the original's letters and stay in the output keys m_c and m_x.
@dataclass(frozen=True)
class Batch:
    """One batch of item pairs of one cell, with the unpatched runs of both texts."""
    cell: str
    ids: torch.Tensor            # (B, T) token ids of the clean texts
    corrupt_ids: torch.Tensor    # (B, T) token ids of the corrupt texts
    pairing: torch.Tensor        # (B, T) for each clean position the corrupt position that holds the same word piece, or -1
    clean: Run
    corrupt: Run

    def swap(self, corrupt: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """The change that swapping one tensor (B, T, ...) makes: the corrupt run's values on the clean positions, minus the clean values."""
        return onto_clean(corrupt, clean, self.pairing) - clean

    def flat(self, t: torch.Tensor) -> torch.Tensor:
        """(B, T, d) -> (B * T, d), the shape the dictionaries read"""
        B, T, d = self.clean.post8.shape
        return t.reshape(B * T, d)

    def unflat(self, t: torch.Tensor) -> torch.Tensor:
        """(B * T, d) -> (B, T, d)"""
        B, T, d = self.clean.post8.shape
        return t.view(B, T, d)


class ArmMargins:
    """The margins of all arms, keyed '<cell>/<arm>': one (B,) tensor per batch, in the order the arms were measured.
    with_mix9() and with_mlp9() make the patched run of an arm and store its margin; the model and the score are held for these
    two only. A function that computes with the model itself takes it as its own parameter."""

    def __init__(self, model: HookedModel, score: LineBreakScore) -> None:
        self.model, self.score = model, score
        self.by_arm: dict[str, list[torch.Tensor]] = {}

    def add(self, cell: str, arm: str, m: torch.Tensor) -> None:
        self.by_arm.setdefault(f"{cell}/{arm}", []).append(m.float().cpu())

    def with_mix9(self, batch: Batch, new: torch.Tensor, arm: str) -> None:
        """the clean texts with the output of the L9 mixer replaced by `new` (B, T, d)"""
        self.add(batch.cell, arm, self.score.margin(self.model.run(batch.ids, patch_mix9=lambda t: new).hidden))

    def with_mlp9(self, batch: Batch, new: torch.Tensor, arm: str) -> None:
        """the clean texts with the output of the L9 MLP replaced by `new` (B, T, d)"""
        self.add(batch.cell, arm, self.score.margin(self.model.run(batch.ids, patch_mlp9=lambda t: new).hidden))

    def tensors(self) -> dict[str, torch.Tensor]:
        """'margin/<cell>/<arm>' -> (items of the cell,)"""
        return {f"margin/{k}": torch.cat(v) for k, v in self.by_arm.items()}


def measure_mixer_arms(model: HookedModel, arms: ArmMargins, batch: Batch) -> torch.Tensor:
    """The L9 mixer: its whole output swapped (layer_y); only head 10's read swapped (head_o).
    Returns head 10's swapped write (B, T, d); the link arms use it again."""
    clean, corrupt = batch.clean, batch.corrupt
    arms.with_mix9(batch, onto_clean(corrupt.mix9, clean.mix9, batch.pairing), "layer_y")
    d_head = head_swapped_write(model.mixer9, HEAD, clean.mixer_in9, corrupt.mixer_in9, batch.pairing)
    arms.with_mix9(batch, clean.mix9 + d_head, "head_o")
    return d_head


def through_mixer9(model: HookedModel, arms: ArmMargins, batch: Batch, mixer9_clean: torch.Tensor, change: torch.Tensor, arm: str) -> None:
    """One arm: `change` (B, T, d) is added to resid_post.8 and sent through the model's own L9 mixer only (the residual's direct
    path stays clean). mixer9_clean is mixer9_of() of the clean resid_post.8."""
    clean = batch.clean
    arms.with_mix9(batch, clean.mix9 + mixer9_of(model, clean.post8 + change) - mixer9_clean, arm)


def measure_l8_arms(model: HookedModel, arms: ArmMargins, batch: Batch, molt8: PublicMolt,
                    vocab8: StateVocab) -> tuple[torch.Tensor, torch.Tensor]:
    """Five swaps in resid_post.8, each sent through the L9 mixer: all of resid_post.8, the L8 MLP's output, the writes of all
    transforms of the L8 dictionary, transform 189's write, state group 11586's contribution.
    Returns the gates of transform 189 on the clean texts (B, T, 1) and transform 189's swapped write (B, T, d)."""
    clean, corrupt = batch.clean, batch.corrupt
    site8 = "resid_post.8"
    mixer9_clean = mixer9_of(model, clean.post8)
    gate8_c, y8_c = molt8.forward(batch.flat(clean.mid8))
    _, y8_x = molt8.forward(batch.flat(corrupt.mid8))
    w189 = batch.swap(batch.unflat(molt8.transform_write(batch.flat(corrupt.mid8), L8_TRANSFORM)),
                      batch.unflat(molt8.transform_write(batch.flat(clean.mid8), L8_TRANSFORM)))
    through_mixer9(model, arms, batch, mixer9_clean, batch.swap(corrupt.post8, clean.post8), "l8/full_post8")
    through_mixer9(model, arms, batch, mixer9_clean, batch.swap(corrupt.mlp8, clean.mlp8), "l8/mlp8_true")
    through_mixer9(model, arms, batch, mixer9_clean, batch.swap(batch.unflat(y8_x), batch.unflat(y8_c)), "l8/molt8_all")
    through_mixer9(model, arms, batch, mixer9_clean, w189, "l8/t189")                                   # key spells out L8_TRANSFORM
    group_swap = batch.swap(group_contribution(vocab8, corrupt.post8, site8, L8_GROUP),
                            group_contribution(vocab8, clean.post8, site8, L8_GROUP))
    through_mixer9(model, arms, batch, mixer9_clean, group_swap, "l8/g11586")                           # key spells out L8_GROUP
    B, T, _ = clean.post8.shape
    return gate8_c[:, [L8_TRANSFORM]].view(B, T, -1), w189


def frame_energy(w189: torch.Tensor, group_frame: torch.Tensor, random_frames: list[torch.Tensor]) -> dict[str, float]:
    """How much of transform 189's swapped write (B, T, d) lies inside group 11586's frame. Sums of squares over one batch, position 0
    dropped: of the whole write ("all"), of its part inside the group's frame ("group") and inside each random frame ("random<j>").
    measure() adds the batches up and divides by "all"."""
    w = w189[:, 1:]
    energy = {"all": float(w.pow(2).sum()), "group": float(in_frame(group_frame, w).pow(2).sum())}
    for j, frame in enumerate(random_frames):
        energy[f"random{j}"] = float(in_frame(frame, w).pow(2).sum())
    return energy


def measure_l9_arms(arms: ArmMargins, batch: Batch, molt9: PublicMolt) -> torch.Tensor:
    """Four swaps at the L9 MLP's output: the whole output, the writes of all transforms of the L9 dictionary, transform 410's write,
    transform 1749's write. Returns the gates of transforms 410 and 1749 on the clean texts (B, T, 2)."""
    clean, corrupt = batch.clean, batch.corrupt
    gate9_c, y9_c = molt9.forward(batch.flat(clean.mid9))
    _, y9_x = molt9.forward(batch.flat(corrupt.mid9))
    arms.with_mlp9(batch, clean.mlp9 + batch.swap(corrupt.mlp9, clean.mlp9), "l9/mlp9_y")
    arms.with_mlp9(batch, clean.mlp9 + batch.swap(batch.unflat(y9_x), batch.unflat(y9_c)), "l9/molt9_all")
    for t in L9_TRANSFORMS:
        write_c = batch.unflat(molt9.transform_write(batch.flat(clean.mid9), t))
        write_x = batch.unflat(molt9.transform_write(batch.flat(corrupt.mid9), t))
        arms.with_mlp9(batch, clean.mlp9 + batch.swap(write_x, write_c), f"l9/t{t}")
    B, T, _ = clean.post8.shape
    return gate9_c[:, list(L9_TRANSFORMS)].view(B, T, -1)


def link_writes(batch: Batch, molt9: PublicMolt, mid9: torch.Tensor) -> dict[str, torch.Tensor]:
    """For a resid_mid.9 (B, T, d): per route of LINK_ROUTES, the summed writes (B, T, d) of the route's L9 transforms."""
    one = {t: batch.unflat(molt9.transform_write(batch.flat(mid9), t)) for t in LINK_ROUTES["F2"]}
    return {route: sum(one[t] for t in transforms) for route, transforms in LINK_ROUTES.items()}


def link_through_transforms(arms: ArmMargins, batch: Batch, molt9: PublicMolt, writes_clean: dict[str, torch.Tensor],
                            change: torch.Tensor, arm: str) -> None:
    """Arms link/<arm>/F2 and link/<arm>/T2: of a `change` (B, T, d) of the mixer's output, only the change it causes in the writes
    of a route's L9 transforms reaches the answer. writes_clean is link_writes() of the clean resid_mid.9."""
    clean = batch.clean
    changed = link_writes(batch, molt9, clean.mid9 + change)
    for route in LINK_ROUTES:
        arms.with_mlp9(batch, clean.mlp9 + changed[route] - writes_clean[route], f"link/{arm}/{route}")


def link_down_and_through(arms: ArmMargins, batch: Batch, molt9: PublicMolt, writes_clean: dict[str, torch.Tensor],
                          change: torch.Tensor, arm: str) -> None:
    """Arm link/<arm>/down: `change` (B, T, d) is added at the mixer's output and everything downstream follows.
    Then the same change through the routes (link_through_transforms)."""
    arms.with_mix9(batch, batch.clean.mix9 + change, f"link/{arm}/down")
    link_through_transforms(arms, batch, molt9, writes_clean, change, arm)


def measure_link_arms(arms: ArmMargins, batch: Batch, molt9: PublicMolt, group_frame: torch.Tensor, random_frames: list[torch.Tensor],
                      d_head: torch.Tensor) -> None:
    """The link from the L9 mixer to the L9 transforms. The mixer's swap (the whole output: "mix"; head 10's write d_head: "h10o"),
    taken three ways: all of it, only its part inside group 4885's frame, the rest. Each is measured with everything downstream
    (`down`) and through the writes of a set of L9 transforms only (`F2`, `T2`). Last, the whole swap inside each random frame,
    through the transforms only."""
    clean, corrupt = batch.clean, batch.corrupt
    writes_clean = link_writes(batch, molt9, clean.mid9)
    d_mix = batch.swap(corrupt.mix9, clean.mix9)
    for base, change in (("mix", d_mix), ("h10o", d_head)):
        inside = in_frame(group_frame, change)
        link_down_and_through(arms, batch, molt9, writes_clean, change, f"{base}/full")
        link_down_and_through(arms, batch, molt9, writes_clean, inside, f"{base}/g4885")                       # key spells out L9_GROUP
        link_down_and_through(arms, batch, molt9, writes_clean, change - inside, f"{base}/minus_g4885")        # key spells out L9_GROUP
    for j, frame in enumerate(random_frames):
        link_through_transforms(arms, batch, molt9, writes_clean, in_frame(frame, d_mix), f"mix/rand{j}")


# ---------------------------------------------------------------------------------------------- the measurements: all items, summary
def draw_random_frames(d: int, device: torch.device) -> list[torch.Tensor]:
    """N_RANDOM_FRAMES random frames of the residual space, each (FRAME_DIM, d) with orthonormal rows: the comparison for a state
    group's frame. One generator with seed 0, drawn in order; measure() calls this once, before the first batch."""
    generator = torch.Generator().manual_seed(0)
    return [torch.linalg.qr(torch.randn(d, FRAME_DIM, generator=generator))[0].T.to(device) for _ in range(N_RANDOM_FRAMES)]


def unpatched_batch(model: HookedModel, items: dict[str, torch.Tensor], cell: str, start: int) -> Batch:
    """The item pairs start ... start + BATCH - 1 of one cell on the model's device, with both texts run without a patch."""
    device = model.lm.device
    ids = items[f"{cell}/ids"][start:start + BATCH].long().to(device)
    corrupt_ids = items[f"{cell}/cor"][start:start + BATCH].long().to(device)
    pairing = items[f"{cell}/pi"][start:start + BATCH].long().to(device)
    clean = model.run(ids)
    corrupt = model.run(corrupt_ids)
    return Batch(cell, ids, corrupt_ids, pairing, clean, corrupt)


@torch.inference_mode()
def measure(model: HookedModel, score: LineBreakScore, items: dict[str, torch.Tensor], molt8: PublicMolt, molt9: PublicMolt,
            vocab8: StateVocab, vocab9: StateVocab) -> tuple[dict, dict]:
    """All arms on all items, cell by cell in batches of BATCH. Returns the tensors of result.safetensors (margin/<cell>/<arm>,
    gate/l8, gate/l9, label/<name>) and the energy fractions of transform 189's swapped write."""
    frame8, frame9 = vocab8.Q[L8_GROUP], vocab9.Q[L9_GROUP]
    random_frames = draw_random_frames(vocab8.d, model.lm.device)
    arms = ArmMargins(model, score)
    energy = {"all": 0.0, "group": 0.0, **{f"random{j}": 0.0 for j in range(N_RANDOM_FRAMES)}}
    gates: dict[str, list[torch.Tensor]] = {"l8": [], "l9": []}
    labels: dict[str, list[torch.Tensor]] = {}

    for cell in CELLS:
        for start in range(0, items[f"{cell}/ids"].shape[0], BATCH):
            batch = unpatched_batch(model, items, cell, start)
            arms.add(cell, "m_c", score.margin(batch.clean.hidden))
            arms.add(cell, "m_x", score.margin(batch.corrupt.hidden))
            d_head = measure_mixer_arms(model, arms, batch)
            gate8, w189 = measure_l8_arms(model, arms, batch, molt8, vocab8)
            for name, value in frame_energy(w189, frame8, random_frames).items():
                energy[name] += value
            gate9 = measure_l9_arms(arms, batch, molt9)
            measure_link_arms(arms, batch, molt9, frame9, random_frames, d_head)
            gates["l8"].append(gate8.cpu())
            gates["l9"].append(gate9.cpu())
            for name, label in token_labels(batch.ids, score).items():
                labels.setdefault(name, []).append(label)
            print(f"{cell}: items {start}-{start + len(batch.ids) - 1} done", flush=True)

    tensors = arms.tensors()
    tensors["gate/l8"], tensors["gate/l9"] = torch.cat(gates["l8"]), torch.cat(gates["l9"])
    tensors.update({f"label/{k}": torch.cat(v).to(torch.int32) for k, v in labels.items()})
    return tensors, {"energy_fraction_t189_swap": {k: v / energy["all"] for k, v in energy.items() if k != "all"}}


def shares_by_cell(tensors: dict[str, torch.Tensor]) -> dict[str, dict]:
    """cell -> arm -> half ("all", "even", "odd" by item number) -> [share, 2.5 %, 97.5 %]"""
    shares: dict[str, dict] = {}
    for cell in CELLS:
        m_c, m_x = tensors[f"margin/{cell}/m_c"], tensors[f"margin/{cell}/m_x"]
        index = torch.arange(len(m_c))
        halves = {"all": index >= 0, "even": index % 2 == 0, "odd": index % 2 == 1}
        shares[cell] = {key[len(f"margin/{cell}/"):]: {h: share(m_c[sel], m[sel], m_x[sel]) for h, sel in halves.items()}
                        for key, m in tensors.items() if key.startswith(f"margin/{cell}/") and not key.endswith(("/m_c", "/m_x"))}
    return shares


def firing_rates(tensors: dict[str, torch.Tensor]) -> dict[str, dict]:
    """How often L8 transform 189 and L9 transforms 410 and 1749 fire on the clean texts of the count cells: on the first token of a
    line, on line breaks, on other tokens, and on the other tokens by the number of characters in the line so far."""
    n_count = sum(len(tensors[f"margin/{cell}/m_c"]) for cell in COUNT_CELLS)        # the count cells come first
    lab = {k: tensors[f"label/{k}"][:n_count].bool() if k != "chars" else tensors[f"label/{k}"][:n_count]
           for k in ("chars", "newline", "line_first", "known")}
    usable = lab["known"].clone()
    usable[:, 0] = False                                                              # position 0 is dropped, as everywhere
    where = {"line_first": usable & lab["line_first"], "newline": usable & lab["newline"],
             "other": usable & ~lab["line_first"] & ~lab["newline"]}
    fire: dict[str, dict] = {}
    for name, gate in ((f"L8 transform {L8_TRANSFORM}", tensors["gate/l8"][:n_count, :, 0]),
                       *((f"L9 transform {t}", tensors["gate/l9"][:n_count, :, j]) for j, t in enumerate(L9_TRANSFORMS))):
        on = gate > 0
        fire[name] = {k: float(on[sel].float().mean()) for k, sel in where.items()}
        fire[name]["by_chars_in_line"] = {
            f"{lo}-{lo + CHAR_BIN - 1}": float(on[where["other"] & (lab["chars"] >= lo) & (lab["chars"] < lo + CHAR_BIN)].float().mean())
            for lo in range(0, MAX_CHARS, CHAR_BIN)}
    return fire


def summarize(tensors: dict[str, torch.Tensor], info: dict) -> dict:
    """The content of result.json: the shares, the firing rates, and what measure() returned beside the tensors."""
    return {"shares": shares_by_cell(tensors), "firing_rate": firing_rates(tensors), **info}


def print_summary(result: dict) -> None:
    rows = [("L9 mixer, whole output", "layer_y"), (f"L9 mixer, head {HEAD} only", "head_o"),
            ("resid_post.8, whole -> L9 mixer", "l8/full_post8"), ("L8 MLP output -> L9 mixer", "l8/mlp8_true"),
            ("L8 dictionary, all transforms -> L9 mixer", "l8/molt8_all"), (f"L8 transform {L8_TRANSFORM} -> L9 mixer", "l8/t189"),
            (f"state group {L8_GROUP} -> L9 mixer", "l8/g11586"),
            ("L9 MLP output", "l9/mlp9_y"), ("L9 dictionary, all transforms", "l9/molt9_all"),
            *((f"L9 transform {t}", f"l9/t{t}") for t in L9_TRANSFORMS)]
    print("\nshare of the clean/corrupt margin difference carried by each swap (even-numbered items; all items in brackets)")
    print(f"{'':46s}" + "".join(f"{cell:>18s}" for cell in CELLS))
    for name, arm in rows:
        cells = [result["shares"][cell][arm] for cell in CELLS]
        print(f"{name:46s}" + "".join(f"{v['even'][0]:>10.3f} ({v['all'][0]:.3f})" for v in cells))
    print("\nthe link from the L9 mixer to the L9 transforms: share carried through the writes of six L9 transforms / of transforms "
          f"{L9_TRANSFORMS[0]} and {L9_TRANSFORMS[1]} only (even-numbered items; in brackets: the same change with everything downstream)")
    print(f"{'':42s}" + "".join(f"{cell:>24s}" for cell in CELLS))
    for base, label in (("mix", "L9 mixer swap"), ("h10o", f"head {HEAD} swap")):
        for part, what in (("full", "all of it"), ("g4885", f"inside group {L9_GROUP}'s frame"), ("minus_g4885", "the rest")):
            row = ""
            for cell in CELLS:
                six, two, down = (result["shares"][cell][f"link/{base}/{part}/{route}"]["even"][0] for route in ("F2", "T2", "down"))
                row += f"{six:>9.3f} / {two:.3f} ({down:.3f})"
            print(f"{label + ', ' + what:42s}" + row)
    row = ""
    for cell in CELLS:
        six, two = (max(abs(result["shares"][cell][f"link/mix/rand{j}/{route}"]["even"][0]) for j in range(N_RANDOM_FRAMES)) for route in ("F2", "T2"))
        row += f"{six:>9.3f} / {two:.3f}        "
    print(f"{'L9 mixer swap, inside a random frame (max)':42s}" + row)
    fraction = result["energy_fraction_t189_swap"]
    print(f"\nenergy of transform {L8_TRANSFORM}'s swapped write inside the {FRAME_DIM}-dimensional frame of group {L8_GROUP}: {fraction['group']:.3f}"
          f"   (random {FRAME_DIM}-dimensional frames: at most {max(v for k, v in fraction.items() if k.startswith('random')):.3f})")
    print("\nfiring rate on the clean texts of the count cells")
    for name, rates in result["firing_rate"].items():
        print(f"  {name:18s} line-first tokens {rates['line_first']:.3f}   line breaks {rates['newline']:.3f}   other tokens {rates['other']:.3f}")
        print("    other tokens by characters in the line so far: " + "  ".join(f"{k}: {v:.2f}" for k, v in rates["by_chars_in_line"].items()))


# ---------------------------------------------------------------------------------------------- command line
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, help="folder that holds state/L<L>/ and mlp/L<L>/ of the published set")
    ap.add_argument("--model", default=MODELS["qwen3.5-0.8b"].hf_id, help="Hugging Face id or local folder of Qwen3.5-0.8B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--items", default=ITEMS, help="the 3 x 64 item pairs (token ids and pairing)")
    ap.add_argument("--out", default=None, help="folder for result.json (shares, firing rates) and result.safetensors (per-item margins, gates)")
    args = ap.parse_args()
    spec = replace(MODELS["qwen3.5-0.8b"], hf_id=args.model)
    lm = load_language_model(spec, device=args.device)
    model, score = HookedModel(lm), LineBreakScore(lm)
    molt8 = PublicMolt(os.path.join(args.release, "mlp", "L8"), device=args.device)
    molt9 = PublicMolt(os.path.join(args.release, "mlp", "L9"), device=args.device)
    vocab8 = StateVocab(os.path.join(args.release, "state", "L8"), device=args.device)
    vocab9 = StateVocab(os.path.join(args.release, "state", "L9"), device=args.device)
    tensors, info = measure(model, score, load_file(args.items), molt8, molt9, vocab8, vocab9)
    result = summarize(tensors, info)
    print_summary(result)
    if args.out is not None:
        os.makedirs(args.out, exist_ok=True)
        save_file(tensors, os.path.join(args.out, "result.safetensors"))
        with open(os.path.join(args.out, "result.json"), "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
