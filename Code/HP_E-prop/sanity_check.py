#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sanity Check Script for HP E-prop (`train.py`).

1. sine          : 合成正弦波データを用いたモデル単体の Loss 減少確認 (TIMIT データ不要)
2. timit_overfit : TIMIT の最初の1バッチのみを用いた過学習テスト (パイプライン全体確認)

Usage:
    python3 sanity_check.py --mode sine
    python3 sanity_check.py --mode timit_overfit --feature_type mfcc --steps 100
    python3 sanity_check.py --mode sine --c_reg 0.0 --lr 1e-4
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # GUI なしのサーバー環境でも保存できるように設定
import matplotlib.pyplot as plt
import torch

from train import HPEPropConfig, HPEPropNetwork, build_timit_dataloaders


def generate_synthetic_sine_batch(
    batch_size: int,
    num_frames: int,
    n_in: int,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    各次元で周波数・位相を少しずらした合成正弦波バッチを生成する。
    形状:
        features: (batch_size, num_frames, n_in)
        mask:     (batch_size, num_frames) [すべて 1.0]
    """
    t = torch.linspace(0.0, 2.0 * math.pi, steps=num_frames, device=device, dtype=dtype)
    feats = torch.zeros((batch_size, num_frames, n_in), device=device, dtype=dtype)

    for b in range(batch_size):
        batch_phase = b * (math.pi / max(1, batch_size))
        for d in range(n_in):
            freq = 1.0 + 0.25 * (d % 4)
            phase = batch_phase + 2.0 * math.pi * (d / n_in)
            feats[b, :, d] = 0.5 * torch.sin(freq * t + phase)

    mask = torch.ones((batch_size, num_frames), device=device, dtype=dtype)
    return feats, mask


def save_learning_curves(
    history: Dict[str, List[float]],
    f_target_hz: float,
    title: str,
    out_path: str,
) -> None:
    """Loss (E0, E1, Reg) および発火率 (FR L1, FR L2) の推移をプロットして保存する。"""
    steps = list(range(1, len(history["loss_e0"]) + 1))

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # 1. 予測誤差 E0, E1
    axes[0].plot(steps, history["loss_e0"], label="Loss E0 (Input Prediction)")
    axes[0].plot(steps, history["loss_e1"], label="Loss E1 (L1 Prediction)")
    axes[0].set_ylabel("Prediction Loss")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 発火率正則化損失 Reg
    axes[1].plot(steps, history["loss_reg"], color="tab:green", label="Loss Reg")
    axes[1].set_ylabel("Reg Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. 平均発火率 [Hz]
    axes[2].plot(steps, history["fr_l1_hz"], label="FR L1 [Hz]")
    axes[2].plot(steps, history["fr_l2_hz"], label="FR L2 [Hz]")
    axes[2].axhline(f_target_hz, color="red", linestyle="--", linewidth=1.0, label=f"Target ({f_target_hz:.1f} Hz)")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Firing Rate [Hz]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Saved plot] {out_path}")


def run_overfit_loop(
    model: HPEPropNetwork,
    feats: torch.Tensor,
    mask: Optional[torch.Tensor],
    steps: int,
    log_every: int,
    exp_name: str,
    plot_path: str,
) -> None:
    """同一バッチに対して指定ステップ数の学習を行い、Loss と発火率の推移を検証する。"""
    history: Dict[str, List[float]] = {
        "loss_e0": [],
        "loss_e1": [],
        "loss_reg": [],
        "fr_l1_hz": [],
        "fr_l2_hz": [],
    }

    print(f"\n=== Running {exp_name} ===")
    print(
        f"Input shape: {tuple(feats.shape)} | Device: {model.cfg.device} | "
        f"L1={model.cfg.n_l1}, L2={model.cfg.n_l2} | "
        f"lr={model.cfg.lr}, c_reg={model.cfg.c_reg}, steps={steps}"
    )

    t_start = time.time()
    for step in range(1, steps + 1):
        out = model.forward_batch(feats, mask=mask, train=True, return_spikes=False)

        e0 = out["loss_e0"].item()
        e1 = out["loss_e1"].item()
        reg = out["loss_reg"].item()
        fr1 = out["fr_l1_hz"].item()
        fr2 = out["fr_l2_hz"].item()

        if any(math.isnan(v) or math.isinf(v) for v in (e0, e1, reg, fr1, fr2)):
            raise RuntimeError(
                f"NaN/Inf detected at step {step}: E0={e0}, E1={e1}, Reg={reg}, FR1={fr1}, FR2={fr2}"
            )

        history["loss_e0"].append(e0)
        history["loss_e1"].append(e1)
        history["loss_reg"].append(reg)
        history["fr_l1_hz"].append(fr1)
        history["fr_l2_hz"].append(fr2)

        if step == 1 or step % log_every == 0 or step == steps:
            elapsed = time.time() - t_start
            sec_per_step = elapsed / step
            print(
                f"  Step {step:04d}/{steps:04d} ({sec_per_step:.3f} s/step) | "
                f"E0: {e0:.5f} | E1: {e1:.5f} | Reg: {reg:.6f} | "
                f"FR L1: {fr1:5.1f} Hz | FR L2: {fr2:5.1f} Hz"
            )

    total_time = time.time() - t_start
    e0_init, e0_final = history["loss_e0"][0], history["loss_e0"][-1]
    e1_init, e1_final = history["loss_e1"][0], history["loss_e1"][-1]
    e0_ratio = (e0_final / e0_init) if e0_init > 0 else float("nan")

    print(f"--- {exp_name} Summary ({total_time:.2f} s total) ---")
    print(f"  Loss E0 : {e0_init:.5f} -> {e0_final:.5f} (ratio: {e0_ratio:.3f})")
    print(f"  Loss E1 : {e1_init:.5f} -> {e1_final:.5f}")
    print(f"  FR L1   : {history['fr_l1_hz'][0]:.1f} Hz -> {history['fr_l1_hz'][-1]:.1f} Hz")
    print(f"  FR L2   : {history['fr_l2_hz'][0]:.1f} Hz -> {history['fr_l2_hz'][-1]:.1f} Hz")

    if e0_final < e0_init:
        print("  [PASS] Loss E0 decreased.")
    else:
        print("  [WARN] Loss E0 did NOT decrease. Try c_reg=0.0, lower lr, or check firing rates.")

    save_learning_curves(
        history=history,
        f_target_hz=model.cfg.f_target_hz,
        title=f"HP E-prop Sanity Check: {exp_name}",
        out_path=plot_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check for HP E-prop.")
    parser.add_argument(
        "--mode",
        type=str,
        default="sine",
        choices=["sine", "timit_overfit", "both"],
        help="Experiment mode: 'sine', 'timit_overfit', or 'both'",
    )
    parser.add_argument(
        "--timit_dir",
        type=str,
        default="/home/keitaro-sunagawa/Master-Research/Code/Predictive_E-prop/mainTask",
        help="Path to processed TIMIT directory",
    )
    parser.add_argument("--feature_type", type=str, default="mfcc", choices=["mfcc", "fbank"])
    parser.add_argument("--n_l1", type=int, default=64, help="Number of L1 neurons")
    parser.add_argument("--n_l2", type=int, default=64, help="Number of L2 neurons")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--sine_frames", type=int, default=40, help="Number of frames for synthetic sine wave")
    parser.add_argument("--steps", type=int, default=150, help="Number of optimization steps")
    parser.add_argument("--log_every", type=int, default=10, help="Logging interval")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--c_reg", type=float, default=1.0, help="Firing rate regularization coefficient")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = HPEPropConfig(
        processed_timit_dir=args.timit_dir,
        feature_type=args.feature_type,
        n_l1=args.n_l1,
        n_l2=args.n_l2,
        batch_size=args.batch_size,
        lr=args.lr,
        c_reg=args.c_reg,
        seed=args.seed,
    )

    if args.mode in ("sine", "both"):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        model_sine = HPEPropNetwork(cfg).to(cfg.device)
        feats_sine, mask_sine = generate_synthetic_sine_batch(
            batch_size=cfg.batch_size,
            num_frames=args.sine_frames,
            n_in=cfg.n_in,
            device=cfg.device,
            dtype=cfg.dtype,
        )
        run_overfit_loop(
            model=model_sine,
            feats=feats_sine,
            mask=mask_sine,
            steps=args.steps,
            log_every=args.log_every,
            exp_name="Synthetic Sine Wave",
            plot_path="sanity_check_sine.png",
        )

    if args.mode in ("timit_overfit", "both"):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        train_loader, _ = build_timit_dataloaders(cfg)
        feats_timit, _labels, mask_timit = next(iter(train_loader))

        model_timit = HPEPropNetwork(cfg).to(cfg.device)
        run_overfit_loop(
            model=model_timit,
            feats=feats_timit,
            mask=mask_timit,
            steps=args.steps,
            log_every=args.log_every,
            exp_name=f"TIMIT 1-Batch Overfit ({cfg.feature_type})",
            plot_path=f"sanity_check_timit_{cfg.feature_type}.png",
        )


if __name__ == "__main__":
    main()