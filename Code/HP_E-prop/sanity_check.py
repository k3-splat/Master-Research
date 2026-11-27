#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sanity Check Script for HP E-prop (`train.py`).

1. sine          : 合成正弦波データを用いたモデル単体の Loss 減少確認 (t_delay ms ごとにオンライン更新・発火率正則化)
2. timit_overfit : TIMIT の最初の1バッチのみを用いた過学習テスト (t_delay ms ごとにオンライン更新・発火率正則化)
※ Predictive E-prop (Tables SI–SIV) にならい、再帰重み W_rec に固定スパースマスク (デフォルト sparsity=0.99) を適用する。

Usage:
    python3 sanity_check.py --mode sine --t_delay 10.0 --rec_sparsity 0.99
    python3 sanity_check.py --mode sine --t_delay 10.0 --c_reg 50.0
    python3 sanity_check.py --mode timit_overfit --feature_type mfcc --t_delay 10.0 --rec_sparsity 0.99 --steps 100
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


@torch.no_grad()
def apply_recurrent_sparsity(
    model: HPEPropNetwork,
    sparsity: float = 0.90,
) -> Dict[str, float]:
    """
    Predictive E-prop (Tables SI–SIV) にならい、再帰重み w_rec_1, w_rec_2 に
    固定のスパース接続マスク (接続確率 p_conn = 1 - sparsity、自己結合は常に 0) を適用する。
    `model.diag_mask_1` / `model.diag_mask_2` にスパースマスクを統合するため、
    順伝播・勾配マスク・重み更新後のゼロ維持 (forward_batch / forward_batch_with_delay 両方)
    のすべてでスパース構造が厳密に保持される。
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"rec_sparsity must be in [0.0, 1.0), got {sparsity}")

    p_conn = 1.0 - sparsity
    eye_1 = torch.eye(model.cfg.n_l1, device=model.w_rec_1.device, dtype=model.w_rec_1.dtype)
    eye_2 = torch.eye(model.cfg.n_l2, device=model.w_rec_2.device, dtype=model.w_rec_2.dtype)
    base_diag_1 = 1.0 - eye_1
    base_diag_2 = 1.0 - eye_2

    if sparsity > 0.0:
        sparse_1 = (torch.rand_like(model.w_rec_1) < p_conn).to(model.w_rec_1.dtype)
        sparse_2 = (torch.rand_like(model.w_rec_2) < p_conn).to(model.w_rec_2.dtype)
        mask_1 = sparse_1 * base_diag_1
        mask_2 = sparse_2 * base_diag_2
    else:
        mask_1 = base_diag_1
        mask_2 = base_diag_2

    model.diag_mask_1.copy_(mask_1)
    model.diag_mask_2.copy_(mask_2)
    model.w_rec_1.mul_(model.diag_mask_1)
    model.w_rec_2.mul_(model.diag_mask_2)

    nnz_1 = int(model.diag_mask_1.sum().item())
    nnz_2 = int(model.diag_mask_2.sum().item())
    total_1 = model.diag_mask_1.numel()
    total_2 = model.diag_mask_2.numel()
    return {
        "nnz_l1": float(nnz_1),
        "total_l1": float(total_1),
        "actual_sparsity_l1": 1.0 - (nnz_1 / max(1, total_1)),
        "nnz_l2": float(nnz_2),
        "total_l2": float(total_2),
        "actual_sparsity_l2": 1.0 - (nnz_2 / max(1, total_2)),
    }


@torch.no_grad()
def forward_batch_with_delay(
    model: HPEPropNetwork,
    features: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    t_delay_ms: float = 10.0,
    train: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Predictive E-prop にならい、t_delay_ms (デフォルト 10 ms) ごとに
    発火率抑制項の計算および重み更新を行うオンライン学習版 forward_batch。
    ニューロン状態と適格性トレース (eps_v, eps_a, e_bar) はシーケンス全体で連続保持する。
    """
    cfg = model.cfg
    repeat_steps = cfg.frame_repeat_steps
    delay_steps = max(1, int(round(t_delay_ms / cfg.dt)))

    x_seq = features.repeat_interleave(repeat_steps, dim=1)  # (B, T_frames * 5, n_in)
    batch_size, seq_len, _ = x_seq.shape

    if mask is not None:
        m_seq = mask.repeat_interleave(repeat_steps, dim=1)
    else:
        m_seq = torch.ones((batch_size, seq_len), device=x_seq.device, dtype=x_seq.dtype)

    valid_steps_total = m_seq.sum().clamp(min=1.0).item()
    valid_steps_per_sample_total = m_seq.sum(dim=1, keepdim=True).clamp(min=1.0)
    num_sequences = mask.size(0) if mask is not None else batch_size

    model.reset_state(batch_size)

    grad_acc = {name: torch.zeros_like(p) for name, p in model.named_parameters()} if train else {}

    # シーケンス全体の統計用
    total_loss_e0 = 0.0
    total_loss_e1 = 0.0
    total_loss_reg = 0.0
    reg_window_count = 0
    total_spike_l1 = torch.zeros((batch_size, cfg.n_l1), device=x_seq.device, dtype=x_seq.dtype)
    total_spike_l2 = torch.zeros((batch_size, cfg.n_l2), device=x_seq.device, dtype=x_seq.dtype)

    # t_delay ウィンドウごとの発火率正則化用
    win_spike_l1 = torch.zeros((batch_size, cfg.n_l1), device=x_seq.device, dtype=x_seq.dtype)
    win_spike_l2 = torch.zeros((batch_size, cfg.n_l2), device=x_seq.device, dtype=x_seq.dtype)
    win_valid_steps = torch.zeros((batch_size, 1), device=x_seq.device, dtype=x_seq.dtype)

    w_rec_1_eff = model.w_rec_1 * model.diag_mask_1
    w_rec_2_eff = model.w_rec_2 * model.diag_mask_2
    f_target = cfg.f_target_per_step

    for t in range(seq_len):
        x_t = x_seq[:, t, :]
        m_t = m_seq[:, t : t + 1]  # (B, 1)

        # 1. 現在時刻 t の発火・疑似微分・平滑化スパイク計算
        z1_t, z1_bar_t = model.l1.compute_spike_and_surrogate()
        z2_t, z2_bar_t = model.l2.compute_spike_and_surrogate()

        total_spike_l1.add_(z1_t * m_t)
        total_spike_l2.add_(z2_t * m_t)
        win_spike_l1.add_(z1_t * m_t)
        win_spike_l2.add_(z2_t * m_t)
        win_valid_steps.add_(m_t)

        # 2. トップダウン予測誤差の計算
        pred_0 = torch.matmul(model.l1.z_bar_prev, model.w_top_0.t())
        e0_t = (x_t - pred_0) * m_t

        pred_1 = torch.matmul(model.l2.z_bar_prev, model.w_top_1.t())
        e1_t = (z1_bar_t - pred_1) * m_t

        total_loss_e0 += 0.5 * e0_t.pow(2).sum().item()
        total_loss_e1 += 0.5 * e1_t.pow(2).sum().item()

        # 3. 適格性トレースの更新 & オンライン勾配蓄積
        if train:
            assert (
                model.et_rec_1 is not None
                and model.et_rec_2 is not None
                and model.et_bot_0 is not None
                and model.et_bot_1 is not None
                and model.et_top_1_self is not None
                and model.et_top_0_upper is not None
                and model.et_top_1_upper is not None
                and model.e0_prev is not None
                and model.e1_prev is not None
            )

            model.et_rec_1.update(
                model.l1.alpha, model.l1.rho, model.l1.beta,
                model.l1.psi_prev, model.l1.psi, model.l1.b_prev, model.l1.z_prev,
                model.l1.z_bar_prev, mask_t=m_t,
            )
            model.et_rec_2.update(
                model.l2.alpha, model.l2.rho, model.l2.beta,
                model.l2.psi_prev, model.l2.psi, model.l2.b_prev, model.l2.z_prev,
                model.l2.z_bar_prev, mask_t=m_t,
            )
            model.et_bot_0.update(
                model.l1.alpha, model.l1.rho, model.l1.beta,
                model.l1.psi_prev, model.l1.psi, model.l1.b_prev, model.l1.z_prev,
                model.e0_prev, mask_t=m_t,
            )
            model.et_bot_1.update(
                model.l2.alpha, model.l2.rho, model.l2.beta,
                model.l2.psi_prev, model.l2.psi, model.l2.b_prev, model.l2.z_prev,
                model.e1_prev, mask_t=m_t,
            )
            model.et_top_1_self.update(
                model.l1.alpha, model.l1.rho, model.l1.beta,
                model.l1.psi_prev, model.l1.psi, model.l1.b_prev, model.l1.z_prev,
                model.l2.z_bar_prev2, mask_t=m_t,
            )
            model.et_top_0_upper.update(
                model.l1.alpha, model.l1.rho, model.l1.beta,
                model.l1.psi_prev, model.l1.psi, model.l1.b_prev, model.l1.z_prev,
                model.l1.z_bar_prev2, mask_t=m_t,
            )
            model.et_top_1_upper.update(
                model.l2.alpha, model.l2.rho, model.l2.beta,
                model.l2.psi_prev, model.l2.psi, model.l2.b_prev, model.l2.z_prev,
                model.l2.z_bar_prev2, mask_t=m_t,
            )

            if t > 0:
                L1_prev = model.e1_prev - torch.matmul(e0_t, model.B_1.t())
                L2_prev = -torch.matmul(e1_t, model.B_2.t())

                grad_acc["w_rec_1"].add_(
                    torch.einsum("bi,bij->ij", L1_prev, model.et_rec_1.e_bar_prev)
                )
                grad_acc["w_rec_2"].add_(
                    torch.einsum("bi,bij->ij", L2_prev, model.et_rec_2.e_bar_prev)
                )
                grad_acc["w_bot_0"].add_(
                    torch.einsum("bi,bij->ij", L1_prev, model.et_bot_0.e_bar_prev)
                )
                grad_acc["w_bot_1"].add_(
                    torch.einsum("bi,bij->ij", L2_prev, model.et_bot_1.e_bar_prev)
                )

                weighted_top0_upper = torch.einsum(
                    "bk,bkj->kj", L1_prev, model.et_top_0_upper.e_bar_prev
                )
                grad_acc["w_top_0"].sub_(torch.matmul(model.w_bot_0.t(), weighted_top0_upper))

                grad_acc["w_top_1"].add_(
                    torch.einsum("bi,bij->ij", L1_prev, model.et_top_1_self.e_bar_prev)
                )
                weighted_top1_upper = torch.einsum(
                    "bk,bkj->kj", L2_prev, model.et_top_1_upper.e_bar_prev
                )
                grad_acc["w_top_1"].sub_(torch.matmul(model.w_bot_1.t(), weighted_top1_upper))

            grad_acc["w_top_0"].sub_(torch.einsum("bi,bj->ij", e0_t, model.l1.z_bar_prev))
            grad_acc["w_top_1"].sub_(torch.einsum("bi,bj->ij", e1_t, model.l2.z_bar_prev))

        # 4. 層内再帰入力とボトムアップ入力の計算
        i_rec_1 = torch.matmul(z1_bar_t, w_rec_1_eff.t())
        i_rec_2 = torch.matmul(z2_bar_t, w_rec_2_eff.t())
        i_bot_1 = torch.matmul(e0_t, model.w_bot_0.t())
        i_bot_2 = torch.matmul(e1_t, model.w_bot_1.t())

        # 5. 次時刻 t+1 への膜電位・不応期・適応閾値の更新
        model.l1.update_membrane_and_adaptation(e_self=e1_t, bottom_input=i_bot_1, rec_input=i_rec_1)
        model.l2.update_membrane_and_adaptation(e_self=None, bottom_input=i_bot_2, rec_input=i_rec_2)

        model.e0_prev = e0_t
        model.e1_prev = e1_t

        # 6. t_delay_ms ごと (または最終ステップ) に発火率正則化を計算して重み更新
        is_update_step = ((t + 1) % delay_steps == 0) or (t == seq_len - 1)
        if is_update_step:
            win_active = (win_valid_steps > 0).to(x_seq.dtype)  # (B, 1)
            active_count = win_active.sum().item()
            win_denom = win_valid_steps.clamp(min=1.0)
            f_av_win_l1 = win_spike_l1 / win_denom
            f_av_win_l2 = win_spike_l2 / win_denom

            if active_count > 0:
                loss_reg_win_l1 = (
                    (0.5 * (f_av_win_l1 - f_target).pow(2).sum(dim=-1, keepdim=True) * win_active)
                    .sum()
                    .item()
                    / active_count
                )
                loss_reg_win_l2 = (
                    (0.5 * (f_av_win_l2 - f_target).pow(2).sum(dim=-1, keepdim=True) * win_active)
                    .sum()
                    .item()
                    / active_count
                )
                total_loss_reg += loss_reg_win_l1 + loss_reg_win_l2
                reg_window_count += 1

            if train:
                assert (
                    model.et_rec_1 is not None
                    and model.et_rec_2 is not None
                    and model.et_bot_0 is not None
                    and model.et_bot_1 is not None
                    and model.et_top_1_self is not None
                    and model.et_top_0_upper is not None
                    and model.et_top_1_upper is not None
                )
                if active_count > 0:
                    if cfg.c_reg > 0.0:
                        L_reg_1 = (cfg.c_reg * (f_av_win_l1 - f_target) / win_denom) * win_active
                        L_reg_2 = (cfg.c_reg * (f_av_win_l2 - f_target) / win_denom) * win_active

                        grad_acc["w_rec_1"].add_(torch.einsum("bi,bij->ij", L_reg_1, model.et_rec_1.e_sum))
                        grad_acc["w_rec_2"].add_(torch.einsum("bi,bij->ij", L_reg_2, model.et_rec_2.e_sum))
                        grad_acc["w_bot_0"].add_(torch.einsum("bi,bij->ij", L_reg_1, model.et_bot_0.e_sum))
                        grad_acc["w_bot_1"].add_(torch.einsum("bi,bij->ij", L_reg_2, model.et_bot_1.e_sum))

                        weighted_reg_top0 = torch.einsum("bk,bkj->kj", L_reg_1, model.et_top_0_upper.e_sum)
                        grad_acc["w_top_0"].sub_(torch.matmul(model.w_bot_0.t(), weighted_reg_top0))

                        grad_acc["w_top_1"].add_(torch.einsum("bi,bij->ij", L_reg_1, model.et_top_1_self.e_sum))
                        weighted_reg_top1 = torch.einsum("bk,bkj->kj", L_reg_2, model.et_top_1_upper.e_sum)
                        grad_acc["w_top_1"].sub_(torch.matmul(model.w_bot_1.t(), weighted_reg_top1))

                    model.optimizer.zero_grad()
                    for name, p in model.named_parameters():
                        g = grad_acc[name] / num_sequences
                        if name == "w_rec_1":
                            g = g * model.diag_mask_1
                        elif name == "w_rec_2":
                            g = g * model.diag_mask_2
                        p.grad = g

                    if cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                    model.optimizer.step()

                    # 対角成分およびスパース非接続成分を厳密に 0 に維持し、更新後の再帰重みを反映
                    model.w_rec_1.mul_(model.diag_mask_1)
                    model.w_rec_2.mul_(model.diag_mask_2)
                    w_rec_1_eff = model.w_rec_1 * model.diag_mask_1
                    w_rec_2_eff = model.w_rec_2 * model.diag_mask_2

                # 次の t_delay ウィンドウに向けて勾配・e_sum をリセット
                for g_buf in grad_acc.values():
                    g_buf.zero_()
                for et in (
                    model.et_rec_1,
                    model.et_rec_2,
                    model.et_bot_0,
                    model.et_bot_1,
                    model.et_top_1_self,
                    model.et_top_0_upper,
                    model.et_top_1_upper,
                ):
                    et.e_sum.zero_()

            # ウィンドウ内のスパイク統計をリセット
            win_spike_l1.zero_()
            win_spike_l2.zero_()
            win_valid_steps.zero_()

    f_av_l1_total = total_spike_l1 / valid_steps_per_sample_total
    f_av_l2_total = total_spike_l2 / valid_steps_per_sample_total
    fr_l1_hz = f_av_l1_total.mean().item() * (1000.0 / cfg.dt)
    fr_l2_hz = f_av_l2_total.mean().item() * (1000.0 / cfg.dt)

    return {
        "loss_e0": torch.tensor(total_loss_e0 / valid_steps_total),
        "loss_e1": torch.tensor(total_loss_e1 / valid_steps_total),
        "loss_reg": torch.tensor(total_loss_reg / max(1, reg_window_count)),
        "fr_l1_hz": torch.tensor(fr_l1_hz),
        "fr_l2_hz": torch.tensor(fr_l2_hz),
    }


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
    t_delay_ms: Optional[float] = None,
    rec_sparsity: float = 0.0,
    sparsity_stats: Optional[Dict[str, float]] = None,
) -> None:
    """同一バッチに対して指定ステップ数の学習を行い、Loss と発火率の推移を検証する。"""
    history: Dict[str, List[float]] = {
        "loss_e0": [],
        "loss_e1": [],
        "loss_reg": [],
        "fr_l1_hz": [],
        "fr_l2_hz": [],
    }

    update_info = f"t_delay={t_delay_ms}ms" if (t_delay_ms is not None and t_delay_ms > 0) else "end-of-batch"
    print(f"\n=== Running {exp_name} ===")
    print(
        f"Input shape: {tuple(feats.shape)} | Device: {model.cfg.device} | "
        f"L1={model.cfg.n_l1}, L2={model.cfg.n_l2} | "
        f"lr={model.cfg.lr}, c_reg={model.cfg.c_reg}, rec_sparsity={rec_sparsity:.3f}, "
        f"update={update_info}, steps={steps}"
    )
    if sparsity_stats is not None:
        print(
            f"Recurrent connectivity | "
            f"L1: {int(sparsity_stats['nnz_l1'])}/{int(sparsity_stats['total_l1'])} "
            f"(actual sparsity={sparsity_stats['actual_sparsity_l1']:.4f}) | "
            f"L2: {int(sparsity_stats['nnz_l2'])}/{int(sparsity_stats['total_l2'])} "
            f"(actual sparsity={sparsity_stats['actual_sparsity_l2']:.4f})"
        )

    t_start = time.time()
    for step in range(1, steps + 1):
        if t_delay_ms is not None and t_delay_ms > 0:
            out = forward_batch_with_delay(model, feats, mask=mask, t_delay_ms=t_delay_ms, train=True)
        else:
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
        title=f"HP E-prop Sanity Check: {exp_name} ({update_info}, sparsity={rec_sparsity:g})",
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
        default="/home/keitaro-sunagawa/Master-Research/TIMIT",
        help="Path to processed TIMIT directory",
    )
    parser.add_argument("--feature_type", type=str, default="mfcc", choices=["mfcc", "fbank"])
    parser.add_argument("--n_l1", type=int, default=64, help="Number of L1 neurons")
    parser.add_argument("--n_l2", type=int, default=64, help="Number of L2 neurons")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--sine_frames", type=int, default=40, help="Number of frames for synthetic sine wave")
    parser.add_argument(
        "--t_delay",
        type=float,
        default=10.0,
        help="Online update & firing-rate reg interval [ms] for sine & timit_overfit modes (<=0 for end-of-batch)",
    )
    parser.add_argument(
        "--rec_sparsity",
        type=float,
        default=0.99,
        help="Sparsity of recurrent weights W_rec following Predictive E-prop (0.99 = 1%% connectivity, 0.0 = dense)",
    )
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
        sparsity_stats_sine = apply_recurrent_sparsity(model_sine, sparsity=args.rec_sparsity)
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
            plot_path="sanity_check_sine10ms1.png",
            t_delay_ms=args.t_delay,
            rec_sparsity=args.rec_sparsity,
            sparsity_stats=sparsity_stats_sine,
        )

    if args.mode in ("timit_overfit", "both"):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        train_loader, _ = build_timit_dataloaders(cfg)
        feats_timit, _labels, mask_timit = next(iter(train_loader))

        model_timit = HPEPropNetwork(cfg).to(cfg.device)
        sparsity_stats_timit = apply_recurrent_sparsity(model_timit, sparsity=args.rec_sparsity)
        timit_plot_path = (
            f"sanity_check_timit_{cfg.feature_type}_{args.t_delay:g}ms.png"
            if args.t_delay > 0
            else f"sanity_check_timit_{cfg.feature_type}.png"
        )
        run_overfit_loop(
            model=model_timit,
            feats=feats_timit,
            mask=mask_timit,
            steps=args.steps,
            log_every=args.log_every,
            exp_name=f"TIMIT 1-Batch Overfit ({cfg.feature_type})",
            plot_path=timit_plot_path,
            t_delay_ms=args.t_delay,
            rec_sparsity=args.rec_sparsity,
            sparsity_stats=sparsity_stats_timit,
        )


if __name__ == "__main__":
    main()