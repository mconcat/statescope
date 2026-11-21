"""The width memory of Qwen3.5-0.8B at layer 12: one GDN head's memory read, or one state group's code, is moved from wrapping
width 100 to width 40, and the line-break margin first reaches the level -3 about 50 characters earlier in the line.

Task. A text is one paragraph wrapped greedily at a fixed width: a word goes on the current line if the line then holds at most that
many characters, so every line break comes from the wrap. To break a line at the right word the model needs two things, how many
characters the line already holds (the count) and how wide the text is (the width). The same documents are wrapped at 40 and at 100
characters; every text is 512 tokens.

Score. m = logit of the line-break token - logsumexp(logits of tokens that start with a space), read at every word-final token that
is not a line break, in lines 3 and later of a text (the width has to be inferred from earlier lines), position 0 excluded. This
is not the score of demo/line_count.py, which reads the last token only and sums over all line-break tokens.
The count of a token is the number of characters of its line up to and including the token. The curve is the mean margin by count
over the 96 test texts wrapped at 100, at counts with at least 20 tokens, smoothed over 3 neighbouring counts (end points repeated).
The crossing is where the smoothed curve first goes from below -3 to at least -3 at a count of 10 or more, interpolated linearly.
Beside it: the mean margin at counts 48 to 50 (a line of a width-40 text has ended; a line of a width-100 text is half full), and
the mean margin at the tokens that a line break follows.

Edits. Each edit is a vector added to the output of the layer-12 mixer of the test texts wrapped at 100 (rounded to the model's
dtype first); everything downstream follows. resid_mid.12 is the layer's input plus this output, so the edit moves it by the same
vector.
  head edit      GDN head 13 of layer 12 writes n_t * w * silu(z_t) through its columns of the output projection: n_t is its memory
                 read divided by its RMS, w the weight of the head norm, z_t the gate of the current token. The read is moved by
                 d = (mean of n at width 40) - (mean of n at width 100), so the write changes by (d * w * silu(z_t)) @ columns, at
                 every position. The means are taken over the tokens inside a line (lines 3 and later; no line break, no first
                 token of a line) of the first 128 documents.
  permuted       the same with the 128 channels of d permuted: a change of the same kind through the same gate that belongs to no
                 width.
  inside frame   only the part of the head edit's change inside the 4-dimensional frame of state group 15458 of resid_mid.12
                 (orthogonal projection in the raw residual).
  rest           the head edit's change minus that part.
  state edit     no head is involved. At every token of lines 3 and later that is not a line break, with count c from 1 to 40, the
                 group's code is moved by (its mean code at count c, width 40) - (its mean code at count c, width 100) and written
                 through the group's frame with the token's own RMS (StateVocab.group_write). A mean code counts the code where the
                 group is among the k active groups and zero where it is not. The means come from 256 documents that are not test
                 documents. A count with fewer than 20 tokens at either width gets no edit: nothing above count 40.
  random         the state edit with, at every count, a vector of the same length along one fixed random direction of the
                 4-dimensional code space.

The head, the group and the level are our study's choices; this script takes them as given and re-measures.
The first 128 documents, from which the mean reads are taken, include the 96 test documents. The head edit acts at every position,
so later layers that look back at earlier tokens see it too. The state edit and the head edit act at different tokens and give
curves of different shape; the table compares where each first reaches the level. The random direction is one draw in a
4-dimensional code space and is not orthogonal to the state edit (its cosine is printed); in our study, of four further
draws the one closest to the state edit's direction (cosine 0.75) also brought the crossing forward, to 37.0.

    python demo/width_memory.py --release <folder with state/ and mlp/> [--device cuda:0] [--out result_folder]

Needs transformers with Qwen3.5 support; on a GPU also flash-linear-attention (without it the GDN layers run a slow reference kernel).

The file has four parts after the constants: the model with hooks and the score, the labels of the texts, layer 12's GDN head by
hand, the measurements.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import types
from dataclasses import dataclass, replace
from typing import Callable

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from statescope.config import MODELS
from statescope.lm import LanguageModel, load_language_model
from statescope.reader import StateVocab

# ---------------------------------------------------------------------------------------------- constants
# LAYER, HEAD, GROUP and LEVEL are our study's choices, not options.
WIDTH, DONOR = 100, 40             # the test texts' width; the width the edits move toward
LAYER, HEAD = 12, 13               # GDN head whose memory read depends on the width
GROUP, SITE = 15458, "resid_mid.12"   # state group whose frame carries the head's write
LEVEL = -3.0                       # the crossing is read at this margin
FIRST_COUNT = 10                   # a crossing is looked for from this count on
MID_COUNTS = (48, 50)              # a width-40 line has ended, a width-100 line is half full
MIN_LINE = 3                       # tokens of lines 0, 1 and 2 are not used
MIN_TOKENS = 20                    # a count needs this many tokens to enter the curve or to be edited
N_COUNTS = 128                     # count axis of every table: 0 ... 127
TEST_DOCS = range(0, 96)           # measured, wrapped at WIDTH
READ_DOCS = range(0, 128)          # the head's mean reads at both widths
CODE_DOCS = range(96, 352)         # the group's mean codes by count at both widths
READ_BATCH, BATCH = 8, 16
PERMUTATION_SEED, DIRECTION_SEED = 0, 2000
ARMS = (("none", "no edit"),
        ("head", f"head {HEAD}'s memory read moved to width {DONOR}"),
        ("head_permuted", "  the same with its channels permuted"),
        ("head_in_frame", f"  only the part inside group {GROUP}'s frame"),
        ("head_rest", "  the rest"),
        ("state", f"group {GROUP}'s code moved to width {DONOR}"),
        ("state_random", "  the same size along a random direction"))
TEXTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "width_memory_texts.safetensors")


# ---------------------------------------------------------------------------------------------- the model with hooks; the score
@dataclass
class Run:
    """One forward pass over a batch (B, T)."""
    hidden: torch.Tensor      # (B, T, d) output of the final norm, model dtype
    mixer_in: torch.Tensor    # what layer 12's mixer receives (after the layer's input norm), model dtype
    mix: torch.Tensor         # the mixer's own output, before any edit, fp32
    resid_mid: torch.Tensor   # resid_mid.12 = what layer 12's second norm receives (with the edit, if there is one), fp32


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


def _add(got: dict[str, torch.Tensor], name: str, edit: torch.Tensor | None) -> Callable:
    """forward hook: store the module's own output as got[name]; with an edit, the module then returns output + edit, the edit
    rounded to the output's dtype before the addition"""
    def hook(module, args, out):
        t = _first(out)
        got[name] = t
        if edit is None:
            return None
        new = t + edit.to(t.dtype)
        return (new, *out[1:]) if isinstance(out, (tuple, list)) else new
    return hook


class HookedModel:
    """The language model with the handles the demonstration needs: layer 12, its mixer and the final norm.

    run() is one forward pass over a batch. It keeps the tensors listed in `Run` and can add an edit (B, T, d) to the output of
    layer 12's mixer on the way; everything after the mixer is then computed from the edited output."""

    def __init__(self, lm: LanguageModel) -> None:
        self.lm = lm
        self.model = lm.model
        self.layer = lm.layers[LAYER]
        self.mixer = self.layer.linear_attn
        self.final_norm = lm.model.model.norm

    @torch.inference_mode()
    def run(self, ids: torch.Tensor, edit: torch.Tensor | None = None) -> Run:
        got: dict[str, torch.Tensor] = {}
        hooks = [self.mixer.register_forward_pre_hook(_keep_input(got, "mixer_in"), with_kwargs=True),
                 self.mixer.register_forward_hook(_add(got, "mix", edit)),
                 self.layer.post_attention_layernorm.register_forward_pre_hook(_keep_input(got, "resid_mid"), with_kwargs=True),
                 self.final_norm.register_forward_hook(_keep(got, "hidden"))]
        try:
            self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
        finally:
            for h in hooks:
                h.remove()
        return Run(got["hidden"], got["mixer_in"], got["mix"].float(), got["resid_mid"].float())


class LineBreakMargin:
    """The score of the task, read from the final hidden state at every token:
    m = logit of the line-break token - logsumexp(logits of tokens that start with a space)."""

    def __init__(self, lm: LanguageModel) -> None:
        unembed = lm.model.get_output_embeddings().weight
        text = [lm.tokenizer.decode([i]) for i in range(unembed.shape[0])]
        self.token_text = text                                             # the text of every token of the vocabulary
        newline = lm.tokenizer("\n", add_special_tokens=False)["input_ids"][0]
        space = [i for i, s in enumerate(text) if s.startswith(" ") and "\n" not in s and "\r" not in s]
        self.w_newline, self.W_space = unembed[newline].float(), unembed[space].float()     # their unembedding rows

    def margin(self, hidden: torch.Tensor) -> torch.Tensor:
        """(B, T, d) final hidden states -> (B, T) margin at every token, on the CPU"""
        h = hidden.float().reshape(-1, hidden.shape[-1])
        m = torch.cat([x @ self.w_newline - (x @ self.W_space.T).logsumexp(1) for x in h.split(1024)])
        return m.view(hidden.shape[:2]).cpu()


# ---------------------------------------------------------------------------------------------- the labels of the texts
@dataclass(frozen=True)
class Texts:
    """The documents wrapped at one width, with the labels of every token; all tensors (documents, T), on the CPU."""
    ids: torch.Tensor          # token ids
    count: torch.Tensor        # characters of the token's line up to and including the token; 0 for a line break
    break_next: torch.Tensor   # the next token is a line break
    eligible: torch.Tensor     # in a line with index >= MIN_LINE, not a line break, not position 0
    inside: torch.Tensor       # eligible and not the first token of its line
    readout: torch.Tensor      # eligible and word-final (the next token starts with a space or is a line break)


def label_texts(ids: torch.Tensor, token_text: list[str]) -> Texts:
    """The labels of ids (documents, T), from the text of the tokens alone."""
    count, break_next, eligible, inside, readout = [], [], [], [], []
    for row in ids.tolist():
        chars, line, after_break = 0, 0, False
        r_count, r_break, r_eligible, r_inside, r_readout = [], [], [], [], []
        for j, token in enumerate(row):
            s = token_text[token]
            following = token_text[row[j + 1]] if j + 1 < len(row) else ""
            is_break = "\n" in s
            if is_break and s != "\n":
                raise ValueError(f"unexpected line-break token {s!r}: the texts hold single line breaks only")
            chars = 0 if is_break else chars + len(s)
            usable = line >= MIN_LINE and not is_break and j > 0
            r_count.append(chars), r_break.append(following.startswith("\n")), r_eligible.append(usable)
            r_inside.append(usable and not after_break)
            r_readout.append(usable and (following.startswith(" ") or following.startswith("\n")))
            line, after_break = line + is_break, is_break
        count.append(r_count), break_next.append(r_break), eligible.append(r_eligible), inside.append(r_inside), readout.append(r_readout)
    return Texts(ids.long(), torch.tensor(count), torch.tensor(break_next), torch.tensor(eligible), torch.tensor(inside), torch.tensor(readout))


# ---------------------------------------------------------------------------------------------- layer 12's GDN head by hand
def _qwen_module() -> types.ModuleType:
    """transformers' Qwen3.5 module: its causal convolution and its l2norm are used below.
    Imported on first use, so that --help works without transformers."""
    from transformers.models.qwen3_5 import modeling_qwen3_5
    return modeling_qwen3_5


def gdn_inputs(mixer: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """The mixer's own projections of its input x (B, T, d), fp32: per head unit q, unit k (B, T, H, dk), v (B, T, H, dv),
    g, beta (B, T, H)."""
    MQ = _qwen_module()
    B, T, _ = x.shape
    mixed = mixer.in_proj_qkv(x).transpose(1, 2)
    b, a = mixer.in_proj_b(x), mixer.in_proj_a(x)
    mixed = MQ.causal_conv1d_fn(mixed, mixer.conv1d.weight.squeeze(1), mixer.conv1d.bias, activation=mixer.activation).transpose(1, 2)
    q, k, v = torch.split(mixed, [mixer.key_dim, mixer.key_dim, mixer.value_dim], dim=-1)
    q, k = q.reshape(B, T, -1, mixer.head_k_dim).float(), k.reshape(B, T, -1, mixer.head_k_dim).float()
    v = v.reshape(B, T, -1, mixer.head_v_dim).float()
    beta = b.float().sigmoid()                                                     # how much of the new value is written
    g = -mixer.A_log.float().exp() * F.softplus(a.float() + mixer.dt_bias.float())  # log of the state's decay
    return {"q": MQ.l2norm(q, dim=-1, eps=1e-6), "k": MQ.l2norm(k, dim=-1, eps=1e-6), "v": v, "g": g, "beta": beta}


def head_read(p: dict[str, torch.Tensor], head: int) -> torch.Tensor:
    """One head's recurrence in fp32, token by token: the state S (dk, dv) decays, the delta rule overwrites what key k_t holds
    with v_t, and the query q_t reads it. Returns the read o (B, T, dv) before the head's norm."""
    k, v, g, beta = p["k"][:, :, head], p["v"][:, :, head], p["g"][:, :, head], p["beta"][:, :, head]
    q = p["q"][:, :, head] / math.sqrt(k.shape[-1])
    B, T, dk = k.shape
    S = torch.zeros(B, dk, v.shape[-1], device=k.device)
    o = torch.zeros_like(v)
    for t in range(T):
        S = S * g[:, t].exp()[:, None, None]
        written = (v[:, t] - torch.einsum("bkv,bk->bv", S, k[:, t])) * beta[:, t, None]
        S = S + k[:, t, :, None] * written[:, None, :]
        o[:, t] = torch.einsum("bkv,bk->bv", S, q[:, t])
    return o


def head_gate(mixer: torch.nn.Module, x: torch.Tensor, head: int) -> torch.Tensor:
    """silu(z_t) of one head for the mixer's input x (B, T, d): (B, T, dv), fp32. The head's write is n_t * w * silu(z_t)."""
    B, T, _ = x.shape
    z = mixer.in_proj_z(x).reshape(B, T, -1, mixer.head_v_dim).float()
    return F.silu(z[:, :, head])


def through_head(mixer: torch.nn.Module, head: int, gate: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """How the mixer's output (B, T, d) changes when the head's normalised read moves by shift (dv,) at every token:
    shift times the head norm's weight and the token's gate (B, T, dv), through the head's columns of the output projection."""
    columns = mixer.out_proj.weight.float().view(mixer.out_proj.weight.shape[0], -1, mixer.head_v_dim)[:, head]
    return (shift * mixer.norm.weight.float() * gate) @ columns.T


# ---------------------------------------------------------------------------------------------- the measurements: the two shifts
@torch.inference_mode()
def mean_head_read(model: HookedModel, texts: Texts) -> torch.Tensor:
    """The head's read divided by its RMS (as the head's norm does), averaged over the inside-a-line tokens of READ_DOCS: (dv,)."""
    device = model.lm.device
    total, n = torch.zeros(model.mixer.head_v_dim, device=device), 0
    for start in range(READ_DOCS.start, READ_DOCS.stop, READ_BATCH):
        rows = slice(start, start + READ_BATCH)
        run = model.run(texts.ids[rows].to(device))
        o = head_read(gdn_inputs(model.mixer, run.mixer_in), HEAD)
        normalised = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + model.mixer.norm.variance_epsilon)
        inside = texts.inside[rows].to(device)
        total += normalised[inside].sum(0)
        n += int(inside.sum())
    return total / n


def group_code(vocab: StateVocab, x: torch.Tensor) -> torch.Tensor:
    """x (B, T, d) raw resid_mid.12 -> the code (B, T, b) of state group GROUP: zero where the group is not among the k active"""
    groups, codes = vocab.encode(x.reshape(-1, x.shape[-1]), SITE)
    return (codes * (groups == GROUP)[:, :, None]).sum(1).view(*x.shape[:2], -1)


@torch.inference_mode()
def mean_code_by_count(model: HookedModel, vocab: StateVocab, texts: Texts) -> tuple[torch.Tensor, torch.Tensor]:
    """Over the eligible tokens of CODE_DOCS: the number of tokens at every count (N_COUNTS,) and the group's mean code there
    (N_COUNTS, b); zero where there is no token. The sums by count are made on the CPU: on a GPU, index_add_ adds in an order that
    changes from run to run."""
    device = model.lm.device
    total, n = torch.zeros(N_COUNTS, vocab.b), torch.zeros(N_COUNTS)
    for start in range(CODE_DOCS.start, CODE_DOCS.stop, BATCH):
        rows = slice(start, start + BATCH)
        run = model.run(texts.ids[rows].to(device))
        eligible = texts.eligible[rows]
        count = texts.count[rows][eligible]
        n.index_add_(0, count, torch.ones_like(count, dtype=torch.float))
        total.index_add_(0, count, group_code(vocab, run.resid_mid).cpu()[eligible])
    return n.to(device), (total / n.clamp_min(1)[:, None]).to(device)


# ---------------------------------------------------------------------------------------------- the measurements: the arms
def random_code_direction(b: int, device: torch.device) -> torch.Tensor:
    """one fixed random unit direction of the b-dimensional code space"""
    u = torch.randn(b, generator=torch.Generator().manual_seed(DIRECTION_SEED), dtype=torch.double)
    return (u / u.norm()).float().to(device)


@torch.inference_mode()
def measure(model: HookedModel, score: LineBreakMargin, vocab: StateVocab, texts: dict[int, Texts]) -> tuple[dict, dict]:
    """Returns the tensors of result.safetensors (the two shifts, the token numbers by count, and for the readout tokens of the
    test texts their count, whether a line break follows and the margin of every arm) and what else result.json holds."""
    device = model.lm.device
    test = texts[WIDTH]
    frame = vocab.Q[GROUP]                                                     # (b, d), orthonormal rows

    # the head's shift: its mean normalised read at the donor width minus at the texts' width
    read_shift = mean_head_read(model, texts[DONOR]) - mean_head_read(model, texts[WIDTH])
    permutation = torch.randperm(len(read_shift), generator=torch.Generator().manual_seed(PERMUTATION_SEED)).to(device)
    print(f"mean reads of head {HEAD} at widths {DONOR} and {WIDTH}: done", flush=True)

    # the group's shift by count: its mean code at the donor width minus at the texts' width, where both widths have enough tokens
    n_donor, code_donor = mean_code_by_count(model, vocab, texts[DONOR])
    n_width, code_width = mean_code_by_count(model, vocab, texts[WIDTH])
    edited = (n_donor >= MIN_TOKENS) & (n_width >= MIN_TOKENS)
    edited[0] = False
    code_shift = (code_donor - code_width) * edited[:, None].float()           # (N_COUNTS, b)
    random_shift = code_shift.norm(dim=-1, keepdim=True) * random_code_direction(vocab.b, device)
    print(f"mean codes of group {GROUP} by count at widths {DONOR} and {WIDTH}: done", flush=True)

    margins: dict[str, list[torch.Tensor]] = {arm: [] for arm, _ in ARMS}
    sizes: dict[str, list[float]] = {arm: [] for arm, _ in ARMS}
    energy = {"all": 0.0, "frame": 0.0}
    for start in range(TEST_DOCS.start, TEST_DOCS.stop, BATCH):
        rows = slice(start, start + BATCH)
        ids, count = test.ids[rows].to(device), test.count[rows].to(device)
        eligible, inside, readout = test.eligible[rows].to(device), test.inside[rows].to(device), test.readout[rows]
        clean = model.run(ids)
        rms = vocab.token_rms(clean.resid_mid.reshape(-1, vocab.d)).view(*ids.shape, 1)      # of the unedited resid_mid.12
        gate = head_gate(model.mixer, clean.mixer_in, HEAD)
        head = through_head(model.mixer, HEAD, gate, read_shift)
        head_in_frame = (head @ frame.T) @ frame
        edits = {"head": head,
                 "head_permuted": through_head(model.mixer, HEAD, gate, read_shift[permutation]),
                 "head_in_frame": head_in_frame,
                 "head_rest": head - head_in_frame,
                 "state": vocab.group_write(GROUP, code_shift[count], SITE, rms) * eligible[..., None].float(),
                 "state_random": vocab.group_write(GROUP, random_shift[count], SITE, rms) * eligible[..., None].float()}
        energy["all"] += float(head.double().pow(2).sum())
        energy["frame"] += float((head @ frame.T).double().pow(2).sum())
        mixer_rms = float(clean.mix[inside].pow(2).mean().sqrt())
        margins["none"].append(score.margin(clean.hidden)[readout])
        sizes["none"].append(0.0)
        for arm, edit in edits.items():
            margins[arm].append(score.margin(model.run(ids, edit=edit).hidden)[readout])
            sizes[arm].append(float(edit[inside].pow(2).mean().sqrt()) / mixer_rms)
        print(f"texts wrapped at {WIDTH}: documents {start}-{start + len(ids) - 1} done", flush=True)

    docs = slice(TEST_DOCS.start, TEST_DOCS.stop)
    tensors = {f"margin/{arm}": torch.cat(m).float().contiguous() for arm, m in margins.items()}
    tensors["count"] = test.count[docs][test.readout[docs]].to(torch.int32)
    tensors["break_next"] = test.break_next[docs][test.readout[docs]].to(torch.uint8)
    tensors["read_shift"], tensors["code_shift"] = read_shift.cpu(), code_shift.cpu()
    tensors[f"tokens_by_count/{DONOR}"], tensors[f"tokens_by_count/{WIDTH}"] = n_donor.cpu(), n_width.cpu()
    info = {"size": {arm: sum(s) / len(s) for arm, s in sizes.items()},
            "edited_counts": [int(edited.nonzero().min()), int(edited.nonzero().max())],
            "share_of_head_edit_inside_frame": energy["frame"] / energy["all"],
            "cosine_of_random_direction_with_state_edit": float((random_shift * code_shift).sum() / (random_shift.norm() * code_shift.norm()))}
    return tensors, info


# ---------------------------------------------------------------------------------------------- the measurements: curve, crossing, summary
def margin_curve(m: torch.Tensor, count: torch.Tensor) -> dict[str, torch.Tensor]:
    """The counts with at least MIN_TOKENS tokens, the number of tokens there, the mean margin, and the mean margin smoothed over
    3 neighbouring counts of the curve (end points repeated)."""
    tokens = torch.bincount(count, minlength=N_COUNTS).double()
    mean = torch.bincount(count, weights=m.double(), minlength=N_COUNTS) / tokens.clamp_min(1)
    counts = (tokens >= MIN_TOKENS).nonzero().squeeze(1)
    y = mean[counts]
    smoothed = torch.stack([torch.cat([y[:1], y[:-1]]), y, torch.cat([y[1:], y[-1:]])]).mean(0)
    return {"count": counts, "tokens": tokens[counts], "mean_margin": y, "mean_margin_smoothed": smoothed}


def crossing(curve: dict[str, torch.Tensor]) -> float | None:
    """The count at which the smoothed curve first goes from below LEVEL to at least LEVEL, between two neighbouring counts of the
    curve from FIRST_COUNT on, interpolated linearly. None if it never does."""
    c, y = curve["count"], curve["mean_margin_smoothed"]
    for i in range(len(c) - 1):
        if c[i] >= FIRST_COUNT and y[i] < LEVEL <= y[i + 1]:
            return float(c[i] + (c[i + 1] - c[i]) * (LEVEL - y[i]) / (y[i + 1] - y[i]))
    return None


def summarize(tensors: dict[str, torch.Tensor], info: dict) -> dict:
    """The content of result.json: per arm the size of the edit, the crossing, the two mean margins and the curve."""
    count, break_next = tensors["count"].long(), tensors["break_next"].bool()
    mid = (count >= MID_COUNTS[0]) & (count <= MID_COUNTS[1])
    arms = {}
    for arm, _ in ARMS:
        m = tensors[f"margin/{arm}"]
        curve = margin_curve(m, count)
        arms[arm] = {"size": info["size"][arm], "crossing": crossing(curve),
                     f"mean_margin_counts_{MID_COUNTS[0]}_{MID_COUNTS[1]}": float(m[mid].mean()),
                     "mean_margin_before_line_break": float(m[break_next].mean()),
                     "curve": {k: v.tolist() for k, v in curve.items()}}
    return {"width": WIDTH, "donor_width": DONOR, "level": LEVEL, "readout_tokens": len(count), "followed_by_line_break": int(break_next.sum()),
            f"tokens_counts_{MID_COUNTS[0]}_{MID_COUNTS[1]}": int(mid.sum()), "edited_counts": info["edited_counts"],
            "share_of_head_edit_inside_frame": info["share_of_head_edit_inside_frame"],
            "cosine_of_random_direction_with_state_edit": info["cosine_of_random_direction_with_state_edit"], "arms": arms}


def print_summary(result: dict) -> None:
    mid = f"mean_margin_counts_{MID_COUNTS[0]}_{MID_COUNTS[1]}"
    print(f"\ntexts wrapped at {WIDTH}: {result['readout_tokens']} word-final tokens, {result['followed_by_line_break']} of them followed by a line break")
    print(f"size = RMS of the edit over RMS of the layer-{LAYER} mixer output")
    print(f"crossing = count at which the smoothed mean margin first reaches {LEVEL:g}")
    print(f"{'':48s}{'size':>7s}{'crossing':>10s}{'margin at ' + str(MID_COUNTS[0]) + '-' + str(MID_COUNTS[1]):>17s}{'margin before a line break':>28s}")
    for arm, name in ARMS:
        a = result["arms"][arm]
        where = "none" if a["crossing"] is None else f"{a['crossing']:.1f}"
        print(f"{name:48s}{a['size']:>7.3f}{where:>10s}{a[mid]:>17.2f}{a['mean_margin_before_line_break']:>28.2f}")
    lo, hi = result["edited_counts"]
    print(f"\nshare of the head edit's squared norm inside the 4-dimensional frame of group {GROUP}: {result['share_of_head_edit_inside_frame']:.3f}")
    print(f"the state edit acts at counts {lo} to {hi}; cosine of the random direction's edit with it: {result['cosine_of_random_direction_with_state_edit']:.2f}")


# ---------------------------------------------------------------------------------------------- command line
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, help="folder that holds state/L<L>/ and mlp/L<L>/ of the published set")
    ap.add_argument("--model", default=MODELS["qwen3.5-0.8b"].hf_id, help="Hugging Face id or local folder of Qwen3.5-0.8B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--texts", default=TEXTS, help="the 352 documents wrapped at 40 and at 100 characters (token ids)")
    ap.add_argument("--out", default=None, help="folder for result.json (the table, the curves) and result.safetensors (per-token margins, the shifts)")
    args = ap.parse_args()
    spec = replace(MODELS["qwen3.5-0.8b"], hf_id=args.model)
    lm = load_language_model(spec, device=args.device)
    model, score = HookedModel(lm), LineBreakMargin(lm)
    vocab = StateVocab(os.path.join(args.release, "state", f"L{LAYER}"), device=args.device)
    ids = load_file(args.texts)
    texts = {width: label_texts(ids[f"width:{width}/ids"], score.token_text) for width in (DONOR, WIDTH)}
    tensors, info = measure(model, score, vocab, texts)
    result = summarize(tensors, info)
    print_summary(result)
    if args.out is not None:
        os.makedirs(args.out, exist_ok=True)
        save_file(tensors, os.path.join(args.out, "result.safetensors"))
        with open(os.path.join(args.out, "result.json"), "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
