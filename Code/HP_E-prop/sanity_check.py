#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sanity Check Script for Multi-layer HP E-prop (`train.py`).

1. sine          : 合成正弦波データを用いたモデル単体の Loss 減少確認
2. timit_overfit : TIMIT の最初の1バッチのみを用いた過学習テスト

Usage:
    python3 sanity_check.py --mode sine --n_layers 6 --n_hidden 128 --t_delay 10.0
    python3 sanity_check.py --mode timit_overfit --feature_type fbank --n_layers 6 --n_hidden 128 --t_delay 10.0 --rec_sparsity 0.90 --c_reg 20.0 --lr 0.0005 --steps 100
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
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
    """各次元で周波数・位相を少しずらした合成正弦波バッチを生成する"""
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
    n_layers: int,
    f_target_hz: float,
    title: str,
    out_path: str,
) -> None:
    """全層の Loss (E0..E_{L-1}, Reg) および発火率 (FR L1..L_L) の推移をプロットして保存する"""
    steps = list(range(1, len(history["loss_e0"]) + 1))

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    # 1. 予測誤差 E0..E_{L-1}
    for k in range(n_layers):
        label = f"Loss E{k} (Input Pred)" if k == 0 else f"Loss E{k} (L{k} Pred)"
        axes[0].plot(steps, history[f"loss_e{k}"], label=label)
    axes[0].set_ylabel("Prediction Loss")
    axes[0].set_title(title)
    axes[0].legend(ncol=min(4, n_layers), fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 2. 発火率正則化損失 Reg
    axes[1].plot(steps, history["loss_reg"], color="tab:green", label="Loss Reg")
    axes[1].set_ylabel("Reg Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. 各層の平均発火率 [Hz]
    for k in range(n_layers):
        axes[2].plot(steps, history[f"fr_l{k + 1}_hz"], label=f"FR L{k + 1} [Hz]")
    axes[2].axhline(f_target_hz, color="red", linestyle="--", linewidth=1.0, label=f"Target ({f_target_hz:.1f} Hz)")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Firing Rate [Hz]")
    axes[2].legend(ncol=min(4, n_layers + 1), fontsize=8)
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
    """同一バッチに対して指定ステップ数の学習を行い、全層の Loss と発火率の推移を検証する"""
    cfg = model.cfg
    L = cfg.n_layers

    history: Dict[str, List[float]] = {"loss_reg": []}
    for k in range(L):
        history[f"loss_e{k}"] = []
        history[f"fr_l{k + 1}_hz"] = []

    update_info = (
        f"t_delay={cfg.update_interval_steps * cfg.dt:g}ms"
        if cfg.update_interval_steps is not None
        else "end-of-batch"
    )

    print(f"\n=== Running {exp_name} ({L} layers) ===")
    print(
        f"Input shape: {tuple(feats.shape)} | Device: {cfg.device} | "
        f"sizes={cfg.layer_sizes} | tau_mem={[round(x, 1) for x in cfg.tau_mems]} | "
        f"tau_r={cfg.tau_r}, tau_d={cfg.tau_d} | "
        f"lr={cfg.lr}, c_reg={cfg.c_reg}, rec_sparsity={cfg.rec_sparsity:.3f}, "
        f"update={update_info}, steps={steps}"
    )

    conn_strs = []
    for k in range(L):
        m = model.get_rec_mask(k)
        nnz = int(m.sum().item())
        tot = m.numel()
        conn_strs.append(f"L{k + 1}:{nnz}/{tot}({1.0 - nnz / max(1, tot):.3f})")
    print("Recurrent connectivity | " + " | ".join(conn_strs))

    t_start = time.time()
    for step in range(1, steps + 1):
        out = model.forward_batch(feats, mask=mask, train=True, return_spikes=False)

        reg = out["loss_reg"].item()
        e_vals = [out[f"loss_e{k}"].item() for k in range(L)]
        fr_vals = [out[f"fr_l{k + 1}_hz"].item() for k in range(L)]

        if any(math.isnan(v) or math.isinf(v) for v in [reg, *e_vals, *fr_vals]):
            raise RuntimeError(f"NaN/Inf detected at step {step}: E={e_vals}, Reg={reg}, FR={fr_vals}")

        history["loss_reg"].append(reg)
        for k in range(L):
            history[f"loss_e{k}"].append(e_vals[k])
            history[f"fr_l{k + 1}_hz"].append(fr_vals[k])

        if step == 1 or step % log_every == 0 or step == steps:
            elapsed = time.time() - t_start
            sec_per_step = elapsed / step
            e_str = " | ".join(f"E{k}:{e_vals[k]:.4f}" for k in range(L))
            fr_str = ", ".join(f"L{k + 1}:{fr_vals[k]:.1f}" for k in range(L))
            print(
                f"  Step {step:04d}/{steps:04d} ({sec_per_step:.3f} s/step) | "
                f"{e_str} | Reg:{reg:.5f} | FR[Hz]({fr_str})"
            )

    total_time = time.time() - t_start
    e0_init, e0_final = history["loss_e0"][0], history["loss_e0"][-1]
    e0_ratio = (e0_final / e0_init) if e0_init > 0 else float("nan")

    print(f"--- {exp_name} Summary ({total_time:.2f} s total) ---")
    print(f"  Loss E0 : {e0_init:.5f} -> {e0_final:.5f} (ratio: {e0_ratio:.3f})")
    for k in range(1, L):
        print(f"  Loss E{k} : {history[f'loss_e{k}'][0]:.5f} -> {history[f'loss_e{k}'][-1]:.5f}")
    fr_summary = ", ".join(
        f"L{k + 1}: {history[f'fr_l{k + 1}_hz'][0]:.1f}->{history[f'fr_l{k + 1}_hz'][-1]:.1f}"
        for k in range(L)
    )
    print(f"  FR [Hz] : {fr_summary}")

    if e0_final < e0_init:
        print("  [PASS] Loss E0 decreased.")
    else:
        print("  [WARN] Loss E0 did NOT decrease. Try adjusting c_reg, lr, or tau_r/tau_d.")

    save_learning_curves(
        history=history,
        n_layers=L,
        f_target_hz=cfg.f_target_hz,
        title=f"HP E-prop Sanity Check: {exp_name} ({L}L, {update_info}, sparsity={cfg.rec_sparsity:g})",
        out_path=plot_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check for Multi-layer HP E-prop.")
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
        default="/home/keitaro-sunagawa/Master-Research/TIMIT",
        help="Path to processed TIMIT directory",
    )
    parser.add_argument("--feature_type", type=str, default="mfcc", choices=["mfcc", "fbank"])
    parser.add_argument("--n_layers", type=int, default=2, help="Number of cortical layers (e.g., 2, 6, 8)")
    parser.add_argument("--n_hidden", type=int, default=128, help="Neurons per cortical layer")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--sine_frames", type=int, default=40, help="Number of frames for synthetic sine wave")
    parser.add_argument(
        "--t_delay",
        type=float,
        default=10.0,
        help="Online update & firing-rate reg interval [ms] (<=0 for end-of-batch)",
    )
    parser.add_argument("--rec_sparsity", type=float, default=0.90, help="Recurrent weight sparsity")
    parser.add_argument("--tau_r", type=float, default=2.0, help="Synaptic filter rise time constant [ms]")
    parser.add_argument("--tau_d", type=float, default=5.0, help="Synaptic filter decay time constant [ms]")
    parser.add_argument("--steps", type=int, default=100, help="Number of optimization steps")
    parser.add_argument("--log_every", type=int, default=10, help="Logging interval")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--c_reg", type=float, default=20.0, help="Firing rate regularization coefficient")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = HPEPropConfig(
        processed_timit_dir=args.timit_dir,
        feature_type=args.feature_type,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        rec_sparsity=args.rec_sparsity,
        update_interval_ms=args.t_delay,
        tau_r=args.tau_r,
        tau_d=args.tau_d,
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
            plot_path=f"sanity_check_sine_{cfg.n_layers}L.png",
        )

    if args.mode in ("timit_overfit", "both"):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        train_loader, _ = build_timit_dataloaders(cfg)
        feats_timit, _labels, mask_timit = next(iter(train_loader))

        model_timit = HPEPropNetwork(cfg).to(cfg.device)
        timit_plot_path = f"sanity_check_timit_{cfg.feature_type}_{cfg.n_layers}L.png"
        run_overfit_loop(
            model=model_timit,
            feats=feats_timit,
            mask=mask_timit,
            steps=args.steps,
            log_every=args.log_every,
            exp_name=f"TIMIT 1-Batch Overfit ({cfg.feature_type})",
            plot_path=timit_plot_path,
        )


if __name__ == "__main__":
    main()