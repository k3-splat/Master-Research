#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External Linear CTC Decoder Evaluation (Linear Probing) for Frozen Multi-layer HP E-prop.

凍結した HP E-prop のスパイク表現から、
  1. 全層結合 (concat_all)
  2. 各層単体 (layer_1, ..., layer_L)
それぞれに対して外付け線形 CTC デコーダを学習し、音素エラー率 (PER) を比較評価する。
TIMIT 標準の 61->39 音素縮約に対応。

Usage:
    # 未学習 (ランダム重み) ベースラインの評価
    python3 eval_ctc.py --feature_type fbank --n_layers 4 --n_hidden 256

    # 学習済みチェックポイントの評価 (39音素縮約あり)
    python3 eval_ctc.py --checkpoint ./checkpoints/.../best_model.pt --feature_type fbank --n_layers 4 --n_hidden 256
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from encoder import TimitProcessedDataset, timit_collate_fn
from train import HPEPropConfig, HPEPropNetwork


# E-prop 公式実装 (timit_processing.py) と同一の TIMIT 61 -> 39 音素縮約テーブル
TIMIT_61_TO_39_TABLE: Dict[str, str] = {
    "ao": "aa",
    "ax": "ah",
    "ax-h": "ah",
    "axr": "er",
    "hv": "hh",
    "ix": "ih",
    "el": "l",
    "em": "m",
    "en": "n",
    "nx": "n",
    "eng": "ng",
    "zh": "sh",
    "pcl": "sil",
    "tcl": "sil",
    "kcl": "sil",
    "bcl": "sil",
    "dcl": "sil",
    "gcl": "sil",
    "h#": "sil",
    "pau": "sil",
    "epi": "sil",
    "ux": "uw",
    "q": "sil",
}


def exp_convolve_spikes(spikes: torch.Tensor, decay: float) -> torch.Tensor:
    """
    スパイク列 (B, T, N) に対して因果的指数平滑化 y^t = decay * y^{t-1} + z^t を適用する。
    (E-prop 公式実装 alif_eligibility_propagation.py の exp_convolve に準拠)
    """
    B, T, N = spikes.shape
    out = torch.empty_like(spikes)
    running = torch.zeros((B, N), device=spikes.device, dtype=spikes.dtype)
    for t in range(T):
        running = running * decay + spikes[:, t, :]
        out[:, t, :] = running
    return out


def frame_labels_to_sequence(label_1d: torch.Tensor, valid_len: int) -> List[int]:
    """
    フレームごとの音素 ID 列 (T_frames,) から有効長分を切り出し、
    連続する同一音素 ID を畳み込んで CTC 用の音素系列ターゲットへ変換する。
    """
    valid_labels = label_1d[:valid_len]
    valid_labels = valid_labels[valid_labels >= 0]
    if valid_labels.numel() == 0:
        return []
    collapsed = torch.unique_consecutive(valid_labels)
    return collapsed.cpu().tolist()


def _find_json_file(timit_dir: str, filename: str) -> Optional[str]:
    """timit_dir 直下および train/develop/test サブディレクトリから指定 JSON ファイルを探索する"""
    candidates = [
        os.path.join(timit_dir, filename),
        os.path.join(timit_dir, "train", filename),
        os.path.join(timit_dir, "develop", filename),
        os.path.join(timit_dir, "test", filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def build_reduction_map_from_phoneme_list(
    phn_list: List[str],
    reduced_phn_list: Optional[List[str]] = None,
) -> List[int]:
    """
    61音素リスト (phonem_list.json) と TIMIT 標準縮約テーブルから
    61 -> 39 音素のインデックス縮約マップを生成する。
    """
    if reduced_phn_list is not None:
        reduced_vocab = list(reduced_phn_list)
    else:
        reduced_vocab = ["sil"]

    reduction_map: List[int] = []
    for phn in phn_list:
        mapped_phn = TIMIT_61_TO_39_TABLE.get(phn, phn)
        if mapped_phn not in reduced_vocab:
            reduced_vocab.append(mapped_phn)
        reduction_map.append(reduced_vocab.index(mapped_phn))
    return reduction_map


def load_phoneme_metadata(
    timit_dir: str,
    enable_39_reduction: bool = True,
) -> Tuple[int, Optional[List[int]], str]:
    """
    前処理済み TIMIT ディレクトリから音素クラス数と 61->39 縮約マップを読み込む。
    reduced_phn_index_mapping.json が存在しない場合は、phonem_list.json と
    TIMIT 標準縮約テーブルから動的に縮約マップを生成する。
    """
    n_classes = 61
    phn_list: Optional[List[str]] = None
    reduced_phn_list: Optional[List[str]] = None
    reduction_map: Optional[List[int]] = None
    reduction_source = "disabled"

    phn_list_path = _find_json_file(timit_dir, "phonem_list.json")
    if phn_list_path is not None:
        with open(phn_list_path, "r", encoding="utf-8") as f:
            phn_list = json.load(f)
            n_classes = len(phn_list)

    if not enable_39_reduction:
        return n_classes, None, "disabled (--no_reduce_39)"

    # 1. 既存の reduced_phn_index_mapping.json があれば優先使用
    map_path = _find_json_file(timit_dir, "reduced_phn_index_mapping.json")
    if map_path is not None:
        with open(map_path, "r", encoding="utf-8") as f:
            reduction_map = json.load(f)
        reduction_source = f"enabled (loaded from {os.path.relpath(map_path, timit_dir)})"
        return n_classes, reduction_map, reduction_source

    # 2. フォールバック: phonem_list.json から TIMIT 標準 61->39 縮約マップを動的生成
    if phn_list is not None:
        reduced_list_path = _find_json_file(timit_dir, "reduced_phonem_list.json")
        if reduced_list_path is not None:
            with open(reduced_list_path, "r", encoding="utf-8") as f:
                reduced_phn_list = json.load(f)
        reduction_map = build_reduction_map_from_phoneme_list(phn_list, reduced_phn_list)
        n_reduced = len(set(reduction_map))
        reduction_source = f"enabled (auto-built from phonem_list.json -> {n_reduced} classes)"
    else:
        reduction_source = "disabled (neither reduced_phn_index_mapping.json nor phonem_list.json found)"

    return n_classes, reduction_map, reduction_source


def levenshtein_distance(seq1: List[int], seq2: List[int]) -> int:
    """2つの整数系列間のレーベンシュタイン距離（編集距離）を計算する"""
    m, n = len(seq1), len(seq2)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        s1_i = seq1[i - 1]
        for j in range(1, n + 1):
            cost = 0 if s1_i == seq2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[n]


def reduce_phoneme_sequence(seq: List[int], reduction_map: Optional[List[int]]) -> List[int]:
    """61音素系列を 39音素へ縮約する (縮約マップがない場合はそのまま返す)"""
    if reduction_map is None:
        return seq
    mapped = [reduction_map[idx] for idx in seq if 0 <= idx < len(reduction_map)]
    return mapped


def ctc_greedy_decode(logits: torch.Tensor, length: int, blank_id: int) -> List[int]:
    """
    単一発話のロジット (T, C) から Greedy CTC デコードを行う。
    連続する同一トークンを畳み込み、blank_id を除去する。
    """
    pred_ids = torch.argmax(logits[:length], dim=-1).cpu().tolist()
    decoded: List[int] = []
    prev = -1
    for p in pred_ids:
        if p != prev and p != blank_id:
            decoded.append(p)
        prev = p
    return decoded


def ctc_beam_search_decode(
    log_probs: torch.Tensor,
    length: int,
    blank_id: int,
    beam_width: int = 10,
) -> List[int]:
    """
    単一発話の対数確率 (T, C) から CTC Prefix Beam Search デコードを行う。
    """
    lp = log_probs[:length].cpu().tolist()
    neg_inf = -1e30

    def log_add(a: float, b: float) -> float:
        if a <= neg_inf:
            return b
        if b <= neg_inf:
            return a
        if a > b:
            return a + math.log1p(math.exp(b - a))
        return b + math.log1p(math.exp(a - b))

    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, neg_inf)}

    for t in range(length):
        probs_t = lp[t]
        next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}

        top_c = sorted(range(len(probs_t)), key=lambda c: probs_t[c], reverse=True)[: beam_width + 2]
        if blank_id not in top_c:
            top_c.append(blank_id)

        for prefix, (p_b, p_nb) in beams.items():
            p_total = log_add(p_b, p_nb)

            for c in top_c:
                p_c = probs_t[c]
                if c == blank_id:
                    nb_b, nb_nb = next_beams.get(prefix, (neg_inf, neg_inf))
                    nb_b = log_add(nb_b, p_total + p_c)
                    next_beams[prefix] = (nb_b, nb_nb)
                else:
                    end_t = prefix[-1] if prefix else None
                    new_prefix = prefix + (c,)
                    if c == end_t:
                        nb_b, nb_nb = next_beams.get(prefix, (neg_inf, neg_inf))
                        nb_nb = log_add(nb_nb, p_nb + p_c)
                        next_beams[prefix] = (nb_b, nb_nb)

                        nnb_b, nnb_nb = next_beams.get(new_prefix, (neg_inf, neg_inf))
                        nnb_nb = log_add(nnb_nb, p_b + p_c)
                        next_beams[new_prefix] = (nnb_b, nnb_nb)
                    else:
                        nnb_b, nnb_nb = next_beams.get(new_prefix, (neg_inf, neg_inf))
                        nnb_nb = log_add(nnb_nb, p_total + p_c)
                        next_beams[new_prefix] = (nnb_b, nnb_nb)

        sorted_beams = sorted(
            next_beams.items(),
            key=lambda kv: log_add(kv[1][0], kv[1][1]),
            reverse=True,
        )
        beams = dict(sorted_beams[:beam_width])

    best_prefix = max(beams.items(), key=lambda kv: log_add(kv[1][0], kv[1][1]))[0]
    return list(best_prefix)


@torch.no_grad()
def extract_frozen_representations(
    model: HPEPropNetwork,
    loader: DataLoader,
    tau_out_ms: float = 3.0,
    split_name: str = "train",
) -> Tuple[List[List[torch.Tensor]], List[List[int]]]:
    """
    凍結した HP E-prop に全発話を通し、各層の平滑化＆ダウンサンプリング済み表現を
    CPU (float16) にキャッシュする。
    """
    cfg = model.cfg
    L = cfg.n_layers
    repeat_steps = cfg.frame_repeat_steps
    decay_out = math.exp(-cfg.dt / tau_out_ms) if tau_out_ms > 0 else 0.0

    layer_feats: List[List[torch.Tensor]] = [[] for _ in range(L)]
    targets: List[List[int]] = []

    t0 = time.time()
    n_batches = len(loader)
    print(f"[Extract:{split_name}] Extracting frozen representations ({n_batches} batches)...")

    for batch_idx, (feats, labels, mask) in enumerate(loader, start=1):
        out = model.forward_batch(feats, mask=mask, train=False, return_spikes=True)
        B = feats.size(0)
        valid_frame_lens = mask.sum(dim=1).long().cpu().tolist()

        layer_ds_list: List[torch.Tensor] = []
        for k in range(L):
            spk_k = out[f"spikes_l{k + 1}"]  # (B, T_frames * repeat_steps, N_k)
            if decay_out > 0.0:
                spk_k = exp_convolve_spikes(spk_k, decay_out)
            T_frames = spk_k.size(1) // repeat_steps
            spk_ds = spk_k[:, : T_frames * repeat_steps, :].view(B, T_frames, repeat_steps, -1).mean(dim=2)
            layer_ds_list.append(spk_ds)

        for b in range(B):
            t_len = max(1, int(valid_frame_lens[b]))
            seq_target = frame_labels_to_sequence(labels[b], t_len)
            if len(seq_target) == 0:
                continue
            targets.append(seq_target)
            for k in range(L):
                layer_feats[k].append(layer_ds_list[k][b, :t_len, :].to(dtype=torch.float16, device="cpu"))

        if batch_idx % 50 == 0 or batch_idx == n_batches:
            print(f"  [Extract:{split_name}] {batch_idx:04d}/{n_batches:04d} ({time.time() - t0:.1f}s)")

    print(f"[Extract:{split_name}] Done: {len(targets)} utterances cached in {time.time() - t0:.1f}s.")
    return layer_feats, targets


class CachedRepresentationDataset(Dataset):
    """キャッシュされた凍結表現 (指定層または全層結合) と CTC ターゲットの Dataset"""

    def __init__(
        self,
        layer_feats: List[List[torch.Tensor]],
        targets: List[List[int]],
        layer_indices: List[int],
    ) -> None:
        self.layer_feats = layer_feats
        self.targets = targets
        self.layer_indices = layer_indices
        self.in_dim = sum(layer_feats[k][0].size(-1) for k in layer_indices)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[int]]:
        if len(self.layer_indices) == 1:
            x = self.layer_feats[self.layer_indices[0]][idx]
        else:
            x = torch.cat([self.layer_feats[k][idx] for k in self.layer_indices], dim=-1)
        return x, self.targets[idx]


def ctc_collate_fn(
    batch: List[Tuple[torch.Tensor, List[int]]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]]]:
    """可変長の凍結表現と CTC ターゲットをバッチ化する"""
    B = len(batch)
    in_dim = batch[0][0].size(-1)
    input_lengths = torch.tensor([item[0].size(0) for item in batch], dtype=torch.long)
    max_t = int(input_lengths.max().item())

    x_pad = torch.zeros((B, max_t, in_dim), dtype=torch.float32)
    targets_flat: List[int] = []
    targets_list: List[List[int]] = []

    for b, (x, tgt) in enumerate(batch):
        x_pad[b, : x.size(0), :] = x.float()
        targets_flat.extend(tgt)
        targets_list.append(tgt)

    targets_tensor = torch.tensor(targets_flat, dtype=torch.long)
    return x_pad, input_lengths, targets_tensor, targets_list


class LinearCTCDecoder(nn.Module):
    """凍結表現の線形分離能を評価するための外付け線形 CTC デコーダ"""

    def __init__(self, in_dim: int, n_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.linear = nn.Linear(in_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(x))


@torch.no_grad()
def evaluate_ctc_decoder(
    decoder: LinearCTCDecoder,
    loader: DataLoader,
    blank_id: int,
    reduction_map: Optional[List[int]],
    device: str,
    beam_width: int = 1,
) -> Tuple[float, float]:
    """CTC Loss と PER [%] (39音素縮約適用) を算出する"""
    decoder.eval()
    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    total_loss = 0.0
    n_batches = 0
    total_edits = 0
    total_ref_len = 0

    for x_pad, input_lengths, targets_flat, targets_list in loader:
        x_pad = x_pad.to(device)
        input_lengths_dev = input_lengths.to(device)
        targets_flat_dev = targets_flat.to(device)
        target_lengths_dev = torch.tensor([len(t) for t in targets_list], dtype=torch.long, device=device)

        logits = decoder(x_pad)  # (B, T, C+1)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = ctc_loss_fn(log_probs.transpose(0, 1), targets_flat_dev, input_lengths_dev, target_lengths_dev)
        total_loss += loss.item()
        n_batches += 1

        for b in range(x_pad.size(0)):
            t_len = int(input_lengths[b].item())
            if beam_width <= 1:
                hyp = ctc_greedy_decode(logits[b], t_len, blank_id)
            else:
                hyp = ctc_beam_search_decode(log_probs[b], t_len, blank_id, beam_width=beam_width)

            ref = targets_list[b]
            hyp_eval = reduce_phoneme_sequence(hyp, reduction_map)
            ref_eval = reduce_phoneme_sequence(ref, reduction_map)

            total_edits += levenshtein_distance(hyp_eval, ref_eval)
            total_ref_len += len(ref_eval)

    avg_loss = total_loss / max(1, n_batches)
    per_pct = 100.0 * total_edits / max(1, total_ref_len)
    return avg_loss, per_pct


def train_and_evaluate_probe(
    probe_name: str,
    layer_indices: List[int],
    train_cache: Tuple[List[List[torch.Tensor]], List[List[int]]],
    dev_cache: Optional[Tuple[List[List[torch.Tensor]], List[List[int]]]],
    test_cache: Optional[Tuple[List[List[torch.Tensor]], List[List[int]]]],
    n_classes: int,
    reduction_map: Optional[List[int]],
    args: argparse.Namespace,
    device: str,
) -> Dict[str, float]:
    """指定された層構成に対して外付け線形 CTC デコーダを学習・評価する"""
    train_ds = CachedRepresentationDataset(train_cache[0], train_cache[1], layer_indices)
    train_loader = DataLoader(train_ds, batch_size=args.ctc_batch_size, shuffle=True, collate_fn=ctc_collate_fn)

    dev_loader = (
        DataLoader(
            CachedRepresentationDataset(dev_cache[0], dev_cache[1], layer_indices),
            batch_size=args.ctc_batch_size,
            shuffle=False,
            collate_fn=ctc_collate_fn,
        )
        if dev_cache is not None
        else None
    )
    test_loader = (
        DataLoader(
            CachedRepresentationDataset(test_cache[0], test_cache[1], layer_indices),
            batch_size=args.ctc_batch_size,
            shuffle=False,
            collate_fn=ctc_collate_fn,
        )
        if test_cache is not None
        else None
    )

    blank_id = n_classes
    decoder = LinearCTCDecoder(train_ds.in_dim, n_classes, dropout=args.ctc_dropout).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.ctc_lr, weight_decay=args.ctc_weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.ctc_epochs // 3), gamma=0.3)
    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    print(f"\n--- Probing: {probe_name} (in_dim={train_ds.in_dim}, classes={n_classes}+blank) ---")

    best_dev_per = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, args.ctc_epochs + 1):
        decoder.train()
        tr_loss = 0.0
        n_tr = 0
        for x_pad, input_lengths, targets_flat, targets_list in train_loader:
            x_pad = x_pad.to(device)
            input_lengths_dev = input_lengths.to(device)
            targets_flat_dev = targets_flat.to(device)
            target_lengths_dev = torch.tensor([len(t) for t in targets_list], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = decoder(x_pad)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = ctc_loss_fn(log_probs.transpose(0, 1), targets_flat_dev, input_lengths_dev, target_lengths_dev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item()
            n_tr += 1

        scheduler.step()
        tr_loss /= max(1, n_tr)

        if dev_loader is not None and (epoch == 1 or epoch % args.log_every == 0 or epoch == args.ctc_epochs):
            dev_loss, dev_per = evaluate_ctc_decoder(
                decoder, dev_loader, blank_id, reduction_map, device, beam_width=1
            )
            if dev_per < best_dev_per:
                best_dev_per = dev_per
                best_state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
            print(
                f"  Epoch {epoch:02d}/{args.ctc_epochs:02d} | "
                f"Train CTC: {tr_loss:.4f} | Dev CTC: {dev_loss:.4f} | Dev PER (greedy): {dev_per:.2f}%"
            )
        elif dev_loader is None and (epoch == 1 or epoch % args.log_every == 0 or epoch == args.ctc_epochs):
            print(f"  Epoch {epoch:02d}/{args.ctc_epochs:02d} | Train CTC: {tr_loss:.4f}")

    if best_state is not None:
        decoder.load_state_dict(best_state)

    _, train_per = evaluate_ctc_decoder(decoder, train_loader, blank_id, reduction_map, device, beam_width=1)
    dev_per_final = float("nan")
    if dev_loader is not None:
        _, dev_per_final = evaluate_ctc_decoder(
            decoder, dev_loader, blank_id, reduction_map, device, beam_width=args.beam_width
        )
    test_per_final = float("nan")
    if test_loader is not None:
        _, test_per_final = evaluate_ctc_decoder(
            decoder, test_loader, blank_id, reduction_map, device, beam_width=args.beam_width
        )

    print(
        f"  => [{probe_name}] Train PER: {train_per:.2f}% | "
        f"Dev PER: {dev_per_final:.2f}% | Test PER: {test_per_final:.2f}%"
    )
    return {
        "in_dim": float(train_ds.in_dim),
        "train_per": train_per,
        "dev_per": dev_per_final,
        "test_per": test_per_final,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen HP E-prop with external linear CTC decoder.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to trained HP E-prop checkpoint (.pt). Empty = untrained baseline.",
    )
    parser.add_argument("--timit_dir", type=str, default="/home/keitaro-sunagawa/Master-Research/TIMIT")
    parser.add_argument("--feature_type", type=str, default="fbank", choices=["mfcc", "fbank"])
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--rec_sparsity", type=float, default=0.90)
    parser.add_argument("--tau_r", type=float, default=2.0)
    parser.add_argument("--tau_d", type=float, default=5.0)
    parser.add_argument("--tau_out", type=float, default=3.0, help="Readout exponential smoothing tau [ms]")
    parser.add_argument("--extract_batch_size", type=int, default=16)

    # CTC プロービング設定
    parser.add_argument(
        "--probe",
        type=str,
        default="all_and_each",
        choices=["all_and_each", "concat_all", "each"],
        help="Which representations to probe",
    )
    parser.add_argument("--ctc_epochs", type=int, default=30)
    parser.add_argument("--ctc_batch_size", type=int, default=16)
    parser.add_argument("--ctc_lr", type=float, default=5e-3)
    parser.add_argument("--ctc_weight_decay", type=float, default=1e-5)
    parser.add_argument("--ctc_dropout", type=float, default=0.0)
    parser.add_argument("--beam_width", type=int, default=1, help="1=greedy, >1=CTC prefix beam search (e.g., 10)")
    parser.add_argument("--no_reduce_39", action="store_true", help="Disable 61->39 phoneme reduction for PER")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = HPEPropConfig(
        processed_timit_dir=args.timit_dir,
        feature_type=args.feature_type,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        rec_sparsity=args.rec_sparsity,
        tau_r=args.tau_r,
        tau_d=args.tau_d,
        batch_size=args.extract_batch_size,
        seed=args.seed,
    )
    device = cfg.device

    model = HPEPropNetwork(cfg).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"[Loaded checkpoint] {args.checkpoint}")
    else:
        print("[Info] No checkpoint provided. Evaluating UNTRAINED (random weights) HP E-prop baseline.")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_classes, reduction_map, reduction_status = load_phoneme_metadata(
        args.timit_dir, enable_39_reduction=not args.no_reduce_39
    )
    print(
        f"[Dataset] n_classes={n_classes} | "
        f"39-phoneme reduction={reduction_status} | "
        f"tau_out={args.tau_out}ms | beam_width={args.beam_width}"
    )

    torch_dev = torch.device(device)

    def make_loader(split: str) -> Optional[DataLoader]:
        split_dir = os.path.join(args.timit_dir, split)
        if not os.path.isdir(split_dir):
            return None
        ds = TimitProcessedDataset(split_dir, feature_type=cfg.feature_type)
        return DataLoader(
            ds,
            batch_size=args.extract_batch_size,
            shuffle=False,
            collate_fn=lambda b: timit_collate_fn(b, torch_dev),
        )

    train_loader = make_loader("train_pickle")
    assert train_loader is not None, f"Train directory not found under {args.timit_dir}"
    dev_loader = make_loader("develop_pickle")
    test_loader = make_loader("test_pickle")

    train_cache = extract_frozen_representations(model, train_loader, tau_out_ms=args.tau_out, split_name="train")
    dev_cache = (
        extract_frozen_representations(model, dev_loader, tau_out_ms=args.tau_out, split_name="develop")
        if dev_loader is not None
        else None
    )
    test_cache = (
        extract_frozen_representations(model, test_loader, tau_out_ms=args.tau_out, split_name="test")
        if test_loader is not None
        else None
    )

    L = cfg.n_layers
    probes: List[Tuple[str, List[int]]] = []
    if args.probe in ("all_and_each", "concat_all"):
        probes.append(("concat_all", list(range(L))))
    if args.probe in ("all_and_each", "each"):
        for k in range(L):
            probes.append((f"layer_{k + 1}", [k]))

    summary: Dict[str, Dict[str, float]] = {}
    for probe_name, layer_indices in probes:
        summary[probe_name] = train_and_evaluate_probe(
            probe_name=probe_name,
            layer_indices=layer_indices,
            train_cache=train_cache,
            dev_cache=dev_cache,
            test_cache=test_cache,
            n_classes=n_classes,
            reduction_map=reduction_map,
            args=args,
            device=device,
        )

    print("\n================ CTC Linear Probing Summary ================")
    print(f"{'Probe':<14} | {'Dim':>5} | {'Train PER':>10} | {'Dev PER':>10} | {'Test PER':>10}")
    print("-" * 60)
    for name, res in summary.items():
        print(
            f"{name:<14} | {int(res['in_dim']):>5d} | "
            f"{res['train_per']:>9.2f}% | {res['dev_per']:>9.2f}% | {res['test_per']:>9.2f}%"
        )
    print("============================================================")


if __name__ == "__main__":
    main()