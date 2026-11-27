#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-layer Hierarchical Predictive E-prop (HP E-prop) Full Training Script for TIMIT.
Supports arbitrary cortical layers, timestamped unique save directories,
checkpointing (compatible with `eval_ctc.py`), resume training, LR scheduling,
JSON config/history logging, and learning curve plotting.
Requires `encoder.py` in the same directory or Python path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from encoder import TimitProcessedDataset, timit_collate_fn


def _geomspace(start: float, stop: float, num: int) -> Tuple[float, ...]:
    if num <= 1:
        return (float(start),)
    ratio = (stop / start) ** (1.0 / (num - 1))
    return tuple(float(start * (ratio ** i)) for i in range(num))


def _linspace(start: float, stop: float, num: int) -> Tuple[float, ...]:
    if num <= 1:
        return (float(start),)
    step = (stop - start) / (num - 1)
    return tuple(float(start + step * i) for i in range(num))


@dataclass
class HPEPropConfig:
    """多層 HP E-prop (Hierarchical Predictive E-prop) のハイパーパラメータ管理用 dataclass"""

    # データセット設定 (encoder.py の出力形式に対応)
    processed_timit_dir: str = "/home/keitaro-sunagawa/Master-Research/TIMIT"
    feature_type: str = "fbank"  # "mfcc" (39次元) または "fbank" (123次元)

    # ネットワーク構造 (入力次元, 層数, 各層ニューロン数)
    n_in: int = 123                         # __post_init__ で feature_type に合わせて自動設定
    n_layers: int = 4                       # 皮質層数 L (例: 2, 3, 4, 6, 8)
    n_hidden: int = 256                     # layer_sizes 未指定時の各層ニューロン数
    layer_sizes: Optional[Tuple[int, ...]] = None  # 層ごとのニューロン数を個別指定する場合
    alif_ratio_min: float = 0.5             # 第1層の ALIF 比率
    alif_ratio_max: float = 0.8             # 最上位層の ALIF 比率
    rec_sparsity: float = 0.9               # 再帰結合のスパース率 (0.9 = 90%を0にマスク)

    # 時間・同一入力・不応期・更新間隔設定 [ms]
    dt: float = 1.0
    frame_repeat_ms: float = 5.0            # E-prop にならい 5ms 同一入力を継続
    update_interval_ms: Optional[float] = 10.0  # None: frame_repeat_ms ごと, >0: 指定msごと, <=0: バッチ末尾
    refractory_ms: float = 5.0              # 不応期の長さ [ms] (Predictive E-prop にならい 5ms)

    # ニューロン膜時定数・適応閾値時定数 [ms] (低次→高次へ対数等間隔で時間階層性を形成)
    tau_mem_min: float = 10.0
    tau_mem_max: float = 40.0
    tau_alif_min: float = 100.0
    tau_alif_max: float = 300.0

    # 閾値・適応強度・バイアス・疑似微分パラメータ
    v_th: float = 0.6
    beta_val: float = 0.07
    i_bias: float = 0.02
    gamma_d: float = 0.3

    # シナプス平滑化用 因果的ダブル指数フィルタ時定数 [ms]
    tau_r: float = 2.0
    tau_d: float = 5.0

    # 発火率正則化 (E-prop Supplementary Note 2)
    c_reg: float = 20.0                     # 正則化強度係数 (0.0 で無効化)
    f_target_hz: float = 10.0               # 目標発火率 [Hz]

    # 学習・最適化パラメータ
    batch_size: int = 8
    epochs: int = 20
    lr: float = 5e-4
    lr_decay_every: int = 0                 # >0 で指定エポックごとに lr *= lr_decay (0 で無効)
    lr_decay: float = 0.5
    adam_eps: float = 1e-5                  # E-prop TIMIT にならい 1e-5
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    use_adam: bool = True
    seed: int = 42

    # チェックポイント・ログ保存設定
    save_dir: str = "./checkpoints/hp_eprop"
    save_every: int = 5                     # 指定エポックごとに epoch_XX.pt を保存 (0 で無効)
    resume: str = ""                        # 学習を再開するチェックポイントパス

    # 実行デバイス・データ型
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.feature_type == "mfcc":
            self.n_in = 39
        elif self.feature_type == "fbank":
            self.n_in = 123
        else:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}. Choose 'mfcc' or 'fbank'.")

        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}.")

        if self.layer_sizes is not None:
            if len(self.layer_sizes) != self.n_layers:
                raise ValueError(
                    f"len(layer_sizes) ({len(self.layer_sizes)}) must match n_layers ({self.n_layers})."
                )
        else:
            self.layer_sizes = tuple([self.n_hidden] * self.n_layers)

        if not (0.0 <= self.rec_sparsity < 1.0):
            raise ValueError(f"rec_sparsity must be in [0.0, 1.0), got {self.rec_sparsity}.")

    def to_serializable_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["dtype"] = str(self.dtype)
        d["tau_mems"] = list(self.tau_mems)
        d["tau_alifs"] = list(self.tau_alifs)
        d["alif_ratios"] = list(self.alif_ratios)
        return d

    @property
    def frame_repeat_steps(self) -> int:
        return max(1, int(round(self.frame_repeat_ms / self.dt)))

    @property
    def update_interval_steps(self) -> Optional[int]:
        """None の場合はバッチ末尾更新、正の整数の場合はそのステップ数ごとに更新"""
        if self.update_interval_ms is None:
            return self.frame_repeat_steps
        if self.update_interval_ms <= 0:
            return None
        return max(1, int(round(self.update_interval_ms / self.dt)))

    @property
    def refractory_steps(self) -> int:
        return max(0, int(round(self.refractory_ms / self.dt)))

    @property
    def f_target_per_step(self) -> float:
        return self.f_target_hz * (self.dt / 1000.0)

    @property
    def tau_mems(self) -> Tuple[float, ...]:
        return _geomspace(self.tau_mem_min, self.tau_mem_max, self.n_layers)

    @property
    def tau_alifs(self) -> Tuple[float, ...]:
        return _geomspace(self.tau_alif_min, self.tau_alif_max, self.n_layers)

    @property
    def alif_ratios(self) -> Tuple[float, ...]:
        return _linspace(self.alif_ratio_min, self.alif_ratio_max, self.n_layers)

    @property
    def alphas(self) -> Tuple[float, ...]:
        return tuple(math.exp(-self.dt / tau) for tau in self.tau_mems)

    @property
    def rhos(self) -> Tuple[float, ...]:
        return tuple(math.exp(-self.dt / tau) for tau in self.tau_alifs)

    @property
    def decay_r(self) -> float:
        return math.exp(-self.dt / self.tau_r)

    @property
    def decay_d(self) -> float:
        return math.exp(-self.dt / self.tau_d)

    @property
    def syn_scale(self) -> float:
        return 1.0 / (self.tau_r * self.tau_d)


class EligibilityTrace2D:
    """
    ALIF / LIF 統一の適格性ベクトル (eps_v, eps_a)、適格性トレース e、
    ダブル指数フィルタ済みトレース e_bar、および発火率正則化用のサンプル別累積トレース e_sum を
    GPU 上で保持・更新するクラス。
    形状: (batch_size, n_post, n_pre)
    """

    def __init__(
        self,
        batch_size: int,
        n_post: int,
        n_pre: int,
        cfg: HPEPropConfig,
    ) -> None:
        shape = (batch_size, n_post, n_pre)
        dev, dt = cfg.device, cfg.dtype
        self.eps_v = torch.zeros(shape, device=dev, dtype=dt)
        self.eps_a = torch.zeros(shape, device=dev, dtype=dt)
        self.q_e = torch.zeros(shape, device=dev, dtype=dt)
        self.e_bar = torch.zeros(shape, device=dev, dtype=dt)
        self.e_bar_prev = torch.zeros(shape, device=dev, dtype=dt)
        self.e_sum = torch.zeros(shape, device=dev, dtype=dt)

        self.decay_r = cfg.decay_r
        self.decay_d = cfg.decay_d
        self.syn_scale = cfg.syn_scale

    @torch.no_grad()
    def update(
        self,
        alpha: float,
        rho: float,
        beta: torch.Tensor,         # (1, n_post)
        psi_prev: torch.Tensor,     # (B, n_post)
        psi_curr: torch.Tensor,     # (B, n_post)
        b_prev: torch.Tensor,       # (B, n_post)
        z_prev: torch.Tensor,       # (B, n_post)
        pre_input: torch.Tensor,    # (B, n_pre)
        mask_t: Optional[torch.Tensor] = None,  # (B, 1)
    ) -> torch.Tensor:
        """論文 式 (1), (34)-(41), (50)-(51) に基づく適格性トレースと二重フィルタの更新"""
        self.e_bar_prev.copy_(self.e_bar)

        beta_u = beta.unsqueeze(-1)
        psi_p = psi_prev.unsqueeze(-1)
        psi_c = psi_curr.unsqueeze(-1)
        b_p = b_prev.unsqueeze(-1)
        z_p = z_prev.unsqueeze(-1)
        pre_u = pre_input.unsqueeze(1)  # (B, 1, n_pre)

        # ヤコビアン dh^t / dh^{t-1} の各成分 (式 1 のリセット項 -z^{t-1} b^{t-1} に厳密整合)
        j_vv = alpha - b_p * psi_p
        j_va = beta_u * (psi_p * b_p - z_p)
        j_av = psi_p
        j_aa = rho - beta_u * psi_p

        new_eps_v = j_vv * self.eps_v + j_va * self.eps_a + pre_u
        new_eps_a = j_av * self.eps_v + j_aa * self.eps_a
        self.eps_v.copy_(new_eps_v)
        self.eps_a.copy_(new_eps_a)

        e_curr = psi_c * (self.eps_v - beta_u * self.eps_a)

        if mask_t is not None:
            self.e_sum.add_(e_curr * mask_t.unsqueeze(-1))
        else:
            self.e_sum.add_(e_curr)

        self.e_bar.mul_(self.decay_d).add_(self.q_e)
        self.q_e.mul_(self.decay_r).add_(e_curr, alpha=self.syn_scale)

        return self.e_bar


class HPEPropLayer:
    """HP E-prop の単一皮質層 (ALIF + LIF 混在ニューロン群) の状態と動態を管理するクラス"""

    def __init__(
        self,
        n_neurons: int,
        alif_ratio: float,
        alpha: float,
        rho: float,
        beta_val: float,
        i_bias: float,
        cfg: HPEPropConfig,
    ) -> None:
        self.n = n_neurons
        self.alpha = alpha
        self.rho = rho
        self.i_bias = i_bias
        self.v_th = cfg.v_th
        self.gamma_d = cfg.gamma_d
        self.refractory_steps = cfg.refractory_steps
        self.decay_r = cfg.decay_r
        self.decay_d = cfg.decay_d
        self.syn_scale = cfg.syn_scale
        self.device = cfg.device
        self.dtype = cfg.dtype

        n_alif = int(round(n_neurons * alif_ratio))
        beta_vec = torch.zeros(1, n_neurons, device=self.device, dtype=self.dtype)
        if n_alif > 0:
            beta_vec[:, :n_alif] = beta_val
        self.beta = beta_vec

        self.v: torch.Tensor
        self.a: torch.Tensor
        self.b: torch.Tensor
        self.z: torch.Tensor
        self.psi: torch.Tensor
        self.q_z: torch.Tensor
        self.z_bar: torch.Tensor
        self.z_bar_prev: torch.Tensor
        self.z_bar_prev2: torch.Tensor
        self.ref_count: torch.Tensor

        self.psi_prev: torch.Tensor
        self.b_prev: torch.Tensor
        self.z_prev: torch.Tensor

    @torch.no_grad()
    def reset_state(self, batch_size: int) -> None:
        shape = (batch_size, self.n)
        dev, dt = self.device, self.dtype
        self.v = torch.zeros(shape, device=dev, dtype=dt)
        self.a = torch.zeros(shape, device=dev, dtype=dt)
        self.b = torch.full(shape, self.v_th, device=dev, dtype=dt)
        self.z = torch.zeros(shape, device=dev, dtype=dt)
        self.psi = torch.zeros(shape, device=dev, dtype=dt)
        self.q_z = torch.zeros(shape, device=dev, dtype=dt)
        self.z_bar = torch.zeros(shape, device=dev, dtype=dt)
        self.z_bar_prev = torch.zeros(shape, device=dev, dtype=dt)
        self.z_bar_prev2 = torch.zeros(shape, device=dev, dtype=dt)
        self.ref_count = torch.zeros(shape, device=dev, dtype=torch.int32)

        self.psi_prev = torch.zeros(shape, device=dev, dtype=dt)
        self.b_prev = torch.full(shape, self.v_th, device=dev, dtype=dt)
        self.z_prev = torch.zeros(shape, device=dev, dtype=dt)

    @torch.no_grad()
    def compute_spike_and_surrogate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """現在時刻 t の発火 z^t・疑似微分 psi^t・平滑化スパイク z_bar^t を計算する"""
        self.psi_prev.copy_(self.psi)
        self.b_prev.copy_(self.b)
        self.z_prev.copy_(self.z)
        self.z_bar_prev2.copy_(self.z_bar_prev)
        self.z_bar_prev.copy_(self.z_bar)

        self.b = self.v_th + self.beta * self.a
        in_ref = self.ref_count > 0

        fired = (self.v >= self.b) & (~in_ref)
        self.z = fired.to(self.dtype)

        diff = torch.abs(self.v - self.b) / self.v_th
        psi_raw = (self.gamma_d / self.v_th) * torch.clamp(1.0 - diff, min=0.0)
        self.psi = torch.where(in_ref, torch.zeros_like(psi_raw), psi_raw)

        self.z_bar.mul_(self.decay_d).add_(self.q_z)
        self.q_z.mul_(self.decay_r).add_(self.z, alpha=self.syn_scale)

        return self.z, self.z_bar

    @torch.no_grad()
    def update_membrane_and_adaptation(
        self,
        e_self: Optional[torch.Tensor],
        bottom_input: torch.Tensor,
        rec_input: torch.Tensor,
    ) -> None:
        """
        時刻 t -> t+1 への膜電位 v と適応変数 a の更新 (式 1, 6)。
        Predictive E-prop に従い、発火ステップで式 (1) の減算リセット (-z^t * b^t) を行い、
        不応期中 (ref_count > 0) はリセット後の膜電位に固定する。
        """
        in_ref = self.ref_count > 0
        fired = self.z > 0.5

        v_next = self.alpha * self.v + bottom_input + rec_input + self.i_bias - self.z * self.b
        if e_self is not None:
            v_next = v_next - e_self

        self.v = torch.where(in_ref, self.v, v_next)

        ref_dec = torch.clamp(self.ref_count - 1, min=0)
        self.ref_count = torch.where(
            fired,
            torch.full_like(self.ref_count, self.refractory_steps),
            ref_dec,
        )

        self.a.mul_(self.rho).add_(self.z)


class HPEPropNetwork(nn.Module):
    """
    任意層数 L の階層的予測符号化 SNN (HP E-prop) ネットワーク本体。
    - 層インデックス k = 0, ..., L-1 が論文の皮質層 l = 1, ..., L に対応。
    - 入力層 (l=0) の次元は n_in、各皮質層の次元は cfg.layer_sizes[k]。
    """

    def __init__(self, cfg: HPEPropConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_layers = cfg.n_layers
        assert cfg.layer_sizes is not None
        self.layer_sizes: Tuple[int, ...] = cfg.layer_sizes
        self.dims: Tuple[int, ...] = (cfg.n_in,) + self.layer_sizes

        dev, dt = cfg.device, cfg.dtype
        alphas = cfg.alphas
        rhos = cfg.rhos
        alif_ratios = cfg.alif_ratios

        self.layers: List[HPEPropLayer] = [
            HPEPropLayer(
                n_neurons=self.layer_sizes[k],
                alif_ratio=alif_ratios[k],
                alpha=alphas[k],
                rho=rhos[k],
                beta_val=cfg.beta_val,
                i_bias=cfg.i_bias,
                cfg=cfg,
            )
            for k in range(self.n_layers)
        ]

        self.w_rec = nn.ParameterList()
        self.w_bot = nn.ParameterList()
        self.w_top = nn.ParameterList()

        for k in range(self.n_layers):
            n_lower = self.dims[k]
            n_curr = self.dims[k + 1]

            rec_mask = self._create_rec_mask(n_curr, cfg.rec_sparsity, dev, dt)
            self.register_buffer(f"rec_mask_{k}", rec_mask)

            self.w_rec.append(nn.Parameter(self._init_rec_weight(n_curr, rec_mask, dev, dt)))
            self.w_bot.append(
                nn.Parameter(torch.randn(n_curr, n_lower, device=dev, dtype=dt) / math.sqrt(n_lower))
            )
            self.w_top.append(
                nn.Parameter(torch.randn(n_lower, n_curr, device=dev, dtype=dt) / math.sqrt(n_curr))
            )
            self.register_buffer(
                f"B_{k}",
                torch.randn(n_curr, n_lower, device=dev, dtype=dt) / math.sqrt(n_lower),
            )

        if cfg.use_adam:
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=cfg.lr,
                eps=cfg.adam_eps,
                weight_decay=cfg.weight_decay,
            )
        else:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        self.et_rec: List[EligibilityTrace2D] = []
        self.et_bot: List[EligibilityTrace2D] = []
        self.et_top_self: List[Optional[EligibilityTrace2D]] = []
        self.et_top_upper: List[EligibilityTrace2D] = []
        self.e_prev: List[torch.Tensor] = []

    def get_rec_mask(self, k: int) -> torch.Tensor:
        return getattr(self, f"rec_mask_{k}")

    def get_B(self, k: int) -> torch.Tensor:
        return getattr(self, f"B_{k}")

    @staticmethod
    def _create_rec_mask(n: int, sparsity: float, device: str, dtype: torch.dtype) -> torch.Tensor:
        """対角成分を 0 とし、非対角成分を確率 (1 - sparsity) で 1 とする固定スパースマスクを生成"""
        diag_mask = 1.0 - torch.eye(n, device=device, dtype=dtype)
        if sparsity <= 0.0:
            return diag_mask
        keep_prob = 1.0 - sparsity
        sparse_mask = (torch.rand(n, n, device=device) < keep_prob).to(dtype)
        return sparse_mask * diag_mask

    @staticmethod
    def _init_rec_weight(n: int, rec_mask: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
        w = torch.randn(n, n, device=device, dtype=dtype) / math.sqrt(n)
        w.mul_(rec_mask)
        return w

    @torch.no_grad()
    def reset_state(self, batch_size: int) -> None:
        cfg = self.cfg
        for layer in self.layers:
            layer.reset_state(batch_size)

        self.et_rec = []
        self.et_bot = []
        self.et_top_self = []
        self.et_top_upper = []
        self.e_prev = []

        for k in range(self.n_layers):
            n_lower = self.dims[k]
            n_curr = self.dims[k + 1]
            self.et_rec.append(EligibilityTrace2D(batch_size, n_curr, n_curr, cfg))
            self.et_bot.append(EligibilityTrace2D(batch_size, n_curr, n_lower, cfg))
            self.et_top_self.append(
                EligibilityTrace2D(batch_size, n_lower, n_curr, cfg) if k >= 1 else None
            )
            self.et_top_upper.append(EligibilityTrace2D(batch_size, n_curr, n_curr, cfg))
            self.e_prev.append(torch.zeros(batch_size, n_lower, device=cfg.device, dtype=cfg.dtype))

    @torch.no_grad()
    def forward_batch(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        train: bool = True,
        return_spikes: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        encoder.py の DataLoader 出力形式:
            features: (batch_size, num_frames, n_in)
            mask:     (batch_size, num_frames)  [1.0: 有効, 0.0: パディング]
        を受け取り、各フレームを frame_repeat_steps 連続提示して順伝播・オンライン学習を実行する。
        更新間隔は cfg.update_interval_steps に従う (None ならバッチ末尾で1回更新)。
        """
        cfg = self.cfg
        L = self.n_layers
        repeat_steps = cfg.frame_repeat_steps
        update_steps = cfg.update_interval_steps

        x_seq = features.repeat_interleave(repeat_steps, dim=1)
        batch_size, seq_len, _ = x_seq.shape

        if mask is not None:
            m_seq = mask.repeat_interleave(repeat_steps, dim=1)
        else:
            m_seq = torch.ones((batch_size, seq_len), device=x_seq.device, dtype=x_seq.dtype)

        valid_steps_per_sample = m_seq.sum(dim=1, keepdim=True).clamp(min=1.0)
        valid_steps = m_seq.sum().clamp(min=1.0).item()

        self.reset_state(batch_size)

        grad_acc = {name: torch.zeros_like(p) for name, p in self.named_parameters()} if train else {}

        total_loss_e = [0.0 for _ in range(L)]
        total_loss_reg = 0.0
        reg_window_count = 0

        spike_counts = [
            torch.zeros((batch_size, self.layer_sizes[k]), device=x_seq.device, dtype=x_seq.dtype)
            for k in range(L)
        ]
        win_spikes = [
            torch.zeros((batch_size, self.layer_sizes[k]), device=x_seq.device, dtype=x_seq.dtype)
            for k in range(L)
        ]
        win_valid_steps = torch.zeros((batch_size, 1), device=x_seq.device, dtype=x_seq.dtype)

        spikes_out: Optional[List[torch.Tensor]] = (
            [
                torch.empty((batch_size, seq_len, self.layer_sizes[k]), device=x_seq.device, dtype=x_seq.dtype)
                for k in range(L)
            ]
            if return_spikes
            else None
        )

        w_rec_eff = [self.w_rec[k] * self.get_rec_mask(k) for k in range(L)]
        f_target = cfg.f_target_per_step

        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            m_t = m_seq[:, t : t + 1]  # (B, 1)

            # 1. 全層の発火・疑似微分・平滑化スパイク計算
            z_curr: List[torch.Tensor] = []
            z_bar_curr: List[torch.Tensor] = []
            for k in range(L):
                z_k, z_bar_k = self.layers[k].compute_spike_and_surrogate()
                z_curr.append(z_k)
                z_bar_curr.append(z_bar_k)
                spike_counts[k].add_(z_k * m_t)
                win_spikes[k].add_(z_k * m_t)
                if spikes_out is not None:
                    spikes_out[k][:, t, :] = z_k
            win_valid_steps.add_(m_t)

            # 2. 各層のトップダウン予測誤差 e^{(k), t} (k = 0..L-1)
            e_curr: List[torch.Tensor] = []
            for k in range(L):
                target_k = x_t if k == 0 else z_bar_curr[k - 1]
                pred_k = torch.matmul(self.layers[k].z_bar_prev, self.w_top[k].t())
                e_k = (target_k - pred_k) * m_t
                e_curr.append(e_k)
                total_loss_e[k] += 0.5 * e_k.pow(2).sum().item()

            # 3. 適格性トレースの更新 & ウィンドウ内オンライン勾配蓄積
            if train:
                for k in range(L):
                    layer_k = self.layers[k]
                    self.et_rec[k].update(
                        layer_k.alpha, layer_k.rho, layer_k.beta,
                        layer_k.psi_prev, layer_k.psi, layer_k.b_prev, layer_k.z_prev,
                        layer_k.z_bar_prev, mask_t=m_t,
                    )
                    self.et_bot[k].update(
                        layer_k.alpha, layer_k.rho, layer_k.beta,
                        layer_k.psi_prev, layer_k.psi, layer_k.b_prev, layer_k.z_prev,
                        self.e_prev[k], mask_t=m_t,
                    )
                    self.et_top_upper[k].update(
                        layer_k.alpha, layer_k.rho, layer_k.beta,
                        layer_k.psi_prev, layer_k.psi, layer_k.b_prev, layer_k.z_prev,
                        layer_k.z_bar_prev2, mask_t=m_t,
                    )
                    if k >= 1:
                        et_self_k = self.et_top_self[k]
                        assert et_self_k is not None
                        layer_lower = self.layers[k - 1]
                        et_self_k.update(
                            layer_lower.alpha, layer_lower.rho, layer_lower.beta,
                            layer_lower.psi_prev, layer_lower.psi, layer_lower.b_prev, layer_lower.z_prev,
                            layer_k.z_bar_prev2, mask_t=m_t,
                        )

                if t > 0:
                    L_prev: List[torch.Tensor] = []
                    for k in range(L):
                        fb = torch.matmul(e_curr[k], self.get_B(k).t())
                        if k < L - 1:
                            L_prev.append(self.e_prev[k + 1] - fb)
                        else:
                            L_prev.append(-fb)

                    for k in range(L):
                        grad_acc[f"w_rec.{k}"].add_(
                            torch.einsum("bi,bij->ij", L_prev[k], self.et_rec[k].e_bar_prev)
                        )
                        grad_acc[f"w_bot.{k}"].add_(
                            torch.einsum("bi,bij->ij", L_prev[k], self.et_bot[k].e_bar_prev)
                        )
                        weighted_top_upper = torch.einsum(
                            "bm,bmj->mj", L_prev[k], self.et_top_upper[k].e_bar_prev
                        )
                        grad_acc[f"w_top.{k}"].sub_(torch.matmul(self.w_bot[k].t(), weighted_top_upper))

                        if k >= 1:
                            et_self_k = self.et_top_self[k]
                            assert et_self_k is not None
                            grad_acc[f"w_top.{k}"].add_(
                                torch.einsum("bi,bij->ij", L_prev[k - 1], et_self_k.e_bar_prev)
                            )

                for k in range(L):
                    grad_acc[f"w_top.{k}"].sub_(
                        torch.einsum("bi,bj->ij", e_curr[k], self.layers[k].z_bar_prev)
                    )

            # 4 & 5. 入力電流の計算と次時刻 t+1 への膜電位・不応期・適応閾値の更新
            for k in range(L):
                i_rec_k = torch.matmul(z_bar_curr[k], w_rec_eff[k].t())
                i_bot_k = torch.matmul(e_curr[k], self.w_bot[k].t())
                e_self_k = e_curr[k + 1] if k < L - 1 else None
                self.layers[k].update_membrane_and_adaptation(
                    e_self=e_self_k, bottom_input=i_bot_k, rec_input=i_rec_k
                )

            self.e_prev = e_curr

            # 6. 指定間隔ごと (または最終ステップ) に発火率正則化の計算 & パラメータ更新
            is_update_step = (
                (update_steps is not None and (t + 1) % update_steps == 0)
                or (t == seq_len - 1)
            )
            if is_update_step:
                win_active = (win_valid_steps > 0).to(x_seq.dtype)  # (B, 1)
                num_active = win_active.sum().item()

                if num_active > 0:
                    win_denom = win_valid_steps.clamp(min=1.0)
                    f_av_win = [win_spikes[k] / win_denom for k in range(L)]

                    loss_reg_win = 0.0
                    for k in range(L):
                        reg_k = 0.5 * (f_av_win[k] - f_target).pow(2).sum(dim=-1, keepdim=True)
                        loss_reg_win += (reg_k * win_active).sum().item() / num_active
                    total_loss_reg += loss_reg_win
                    reg_window_count += 1

                    if train:
                        if cfg.c_reg > 0.0:
                            L_reg = [
                                (cfg.c_reg * (f_av_win[k] - f_target) / win_denom) * win_active
                                for k in range(L)
                            ]
                            for k in range(L):
                                grad_acc[f"w_rec.{k}"].add_(
                                    torch.einsum("bi,bij->ij", L_reg[k], self.et_rec[k].e_sum)
                                )
                                grad_acc[f"w_bot.{k}"].add_(
                                    torch.einsum("bi,bij->ij", L_reg[k], self.et_bot[k].e_sum)
                                )
                                weighted_reg_top = torch.einsum(
                                    "bm,bmj->mj", L_reg[k], self.et_top_upper[k].e_sum
                                )
                                grad_acc[f"w_top.{k}"].sub_(torch.matmul(self.w_bot[k].t(), weighted_reg_top))

                                if k >= 1:
                                    et_self_k = self.et_top_self[k]
                                    assert et_self_k is not None
                                    grad_acc[f"w_top.{k}"].add_(
                                        torch.einsum("bi,bij->ij", L_reg[k - 1], et_self_k.e_sum)
                                    )

                        self.optimizer.zero_grad()
                        for name, p in self.named_parameters():
                            g = grad_acc[name] / num_active
                            if name.startswith("w_rec."):
                                layer_idx = int(name.split(".")[1])
                                g = g * self.get_rec_mask(layer_idx)
                            p.grad = g

                        if cfg.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)

                        self.optimizer.step()

                        for k in range(L):
                            mask_k = self.get_rec_mask(k)
                            self.w_rec[k].mul_(mask_k)
                            w_rec_eff[k] = self.w_rec[k] * mask_k

                if train:
                    for g_buf in grad_acc.values():
                        g_buf.zero_()
                    for k in range(L):
                        self.et_rec[k].e_sum.zero_()
                        self.et_bot[k].e_sum.zero_()
                        self.et_top_upper[k].e_sum.zero_()
                        et_self_k = self.et_top_self[k]
                        if et_self_k is not None:
                            et_self_k.e_sum.zero_()

                for k in range(L):
                    win_spikes[k].zero_()
                win_valid_steps.zero_()

        result: Dict[str, torch.Tensor] = {
            "loss_reg": torch.tensor(total_loss_reg / max(1, reg_window_count)),
        }
        for k in range(L):
            result[f"loss_e{k}"] = torch.tensor(total_loss_e[k] / valid_steps)
            f_av_k = spike_counts[k] / valid_steps_per_sample
            result[f"fr_l{k + 1}_hz"] = torch.tensor(f_av_k.mean().item() * (1000.0 / cfg.dt))
            if spikes_out is not None:
                result[f"spikes_l{k + 1}"] = spikes_out[k]

        return result


def build_timit_dataloaders(cfg: HPEPropConfig) -> Tuple[DataLoader, Optional[DataLoader]]:
    """encoder.py で前処理された TIMIT データセットから DataLoader を構築する"""
    device = torch.device(cfg.device)
    train_dir = os.path.join(cfg.processed_timit_dir, "train_pickle")
    dev_dir = os.path.join(cfg.processed_timit_dir, "develop_pickle")

    train_dataset = TimitProcessedDataset(train_dir, feature_type=cfg.feature_type)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: timit_collate_fn(b, device),
    )

    dev_loader: Optional[DataLoader] = None
    if os.path.isdir(dev_dir):
        dev_dataset = TimitProcessedDataset(dev_dir, feature_type=cfg.feature_type)
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=lambda b: timit_collate_fn(b, device),
        )

    return train_loader, dev_loader


def save_checkpoint(
    path: str,
    model: HPEPropNetwork,
    epoch: int,
    best_metric: float,
    cfg: HPEPropConfig,
) -> None:
    """モデル・オプティマイザ・ハイパーパラメータをチェックポイントとして保存する (eval_ctc.py 互換)"""
    payload = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict(),
        "config": cfg.to_serializable_dict(),
    }
    torch.save(payload, path)


def save_training_curves(
    history: Dict[str, List[float]],
    n_layers: int,
    f_target_hz: float,
    out_path: str,
) -> None:
    """Train/Dev の予測誤差・発火率正則化損失・平均発火率の推移をプロット保存する"""
    epochs = history["epoch"]
    if not epochs:
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    # 1. Input prediction error E0 (Train vs Dev)
    axes[0].plot(epochs, history["train_e0"], marker="o", markersize=3, label="Train E0")
    if history.get("dev_e0"):
        axes[0].plot(epochs, history["dev_e0"], marker="s", markersize=3, linestyle="--", label="Dev E0")
    axes[0].set_ylabel("Input Pred Loss (E0)")
    axes[0].set_title(f"HP E-prop Training Curves ({n_layers} layers)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Higher-layer prediction errors E1..E_{L-1} & Reg loss
    for k in range(1, n_layers):
        axes[1].plot(epochs, history[f"train_e{k}"], label=f"Train E{k}")
    axes[1].plot(epochs, history["train_reg"], linestyle=":", color="black", label="Train Reg")
    axes[1].set_ylabel("Higher E / Reg Loss")
    axes[1].legend(ncol=min(4, n_layers), fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 3. Firing rates per layer
    for k in range(n_layers):
        axes[2].plot(epochs, history[f"train_fr_l{k + 1}"], label=f"FR L{k + 1} [Hz]")
    axes[2].axhline(f_target_hz, color="red", linestyle="--", linewidth=1.0, label=f"Target ({f_target_hz:.1f} Hz)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Firing Rate [Hz]")
    axes[2].legend(ncol=min(4, n_layers + 1), fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def train_timit(cfg: HPEPropConfig) -> HPEPropNetwork:
    """TIMIT データセットを用いた多層 HP E-prop の自己教師あり本学習ループ"""
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # 新規学習時は save_dir に実行日時を付与してユニーク化 (--resume 時は既存ディレクトリをそのまま使用)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not cfg.resume:
        base_dir = cfg.save_dir.rstrip("/\\")
        cfg.save_dir = f"{base_dir}_{run_timestamp}"

    os.makedirs(cfg.save_dir, exist_ok=True)

    # 再現用のハイパーパラメータ・実行環境・コマンドを config.json に保存
    config_payload = cfg.to_serializable_dict()
    config_payload["run_timestamp"] = run_timestamp
    config_payload["command"] = " ".join(sys.argv)
    config_payload["torch_version"] = torch.__version__
    config_payload["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        config_payload["cuda_device_name"] = torch.cuda.get_device_name(0)

    config_path = os.path.join(cfg.save_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)

    L = cfg.n_layers
    upd_desc = (
        f"{cfg.update_interval_steps * cfg.dt:g}ms"
        if cfg.update_interval_steps is not None
        else "end-of-batch"
    )
    print(
        f"Initializing {L}-layer HP E-prop on {cfg.device} | "
        f"feature={cfg.feature_type} (dim={cfg.n_in}) | "
        f"sizes={cfg.layer_sizes} | rec_sparsity={cfg.rec_sparsity} | "
        f"tau_mem={[round(x, 1) for x in cfg.tau_mems]} | "
        f"tau_alif={[round(x, 1) for x in cfg.tau_alifs]} | "
        f"tau_r={cfg.tau_r}, tau_d={cfg.tau_d} | "
        f"update={upd_desc} | batch_size={cfg.batch_size} | "
        f"lr={cfg.lr}, c_reg={cfg.c_reg}, f_target={cfg.f_target_hz}Hz | "
        f"save_dir={cfg.save_dir}"
    )

    train_loader, dev_loader = build_timit_dataloaders(cfg)
    model = HPEPropNetwork(cfg).to(cfg.device)

    start_epoch = 1
    best_metric = float("inf")

    history: Dict[str, List[float]] = {
        "epoch": [],
        "lr": [],
        "train_reg": [],
        "dev_reg": [],
        "epoch_sec": [],
    }
    for k in range(L):
        history[f"train_e{k}"] = []
        history[f"train_fr_l{k + 1}"] = []
        history[f"dev_e{k}"] = []
        history[f"dev_fr_l{k + 1}"] = []

    history_path = os.path.join(cfg.save_dir, "history.json")

    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=cfg.device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            model.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", float("inf")))
        if os.path.isfile(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        print(f"[Resumed] {cfg.resume} (starting from epoch {start_epoch}, best_metric={best_metric:.5f})")

    for epoch in range(start_epoch, cfg.epochs + 1):
        t_ep = time.time()

        # 学習率減衰 (指定時)
        if cfg.lr_decay_every > 0 and epoch > 1 and (epoch - 1) % cfg.lr_decay_every == 0:
            for pg in model.optimizer.param_groups:
                pg["lr"] *= cfg.lr_decay
            print(f"[LR Decay] Epoch {epoch:02d} -> lr={model.optimizer.param_groups[0]['lr']:.6g}")

        current_lr = model.optimizer.param_groups[0]["lr"]

        # --- Training ---
        tr_e = [0.0 for _ in range(L)]
        tr_fr = [0.0 for _ in range(L)]
        tr_reg = 0.0
        num_batches = len(train_loader)

        for batch_idx, (feats, _labels, mask) in enumerate(train_loader, start=1):
            out = model.forward_batch(feats, mask=mask, train=True, return_spikes=False)
            for k in range(L):
                tr_e[k] += out[f"loss_e{k}"].item()
                tr_fr[k] += out[f"fr_l{k + 1}_hz"].item()
            tr_reg += out["loss_reg"].item()

            if batch_idx % 50 == 0 or batch_idx == num_batches:
                elapsed = time.time() - t_ep
                sec_per_batch = elapsed / batch_idx
                e_str = " | ".join(f"E{k}:{out[f'loss_e{k}'].item():.4f}" for k in range(L))
                fr_str = ", ".join(f"L{k + 1}:{out[f'fr_l{k + 1}_hz'].item():.1f}" for k in range(L))
                print(
                    f"  Epoch {epoch:02d} [{batch_idx:04d}/{num_batches:04d}] ({sec_per_batch:.2f}s/b) | "
                    f"{e_str} | Reg:{out['loss_reg'].item():.5f} | FR[Hz]({fr_str})"
                )

        tr_e = [v / max(1, num_batches) for v in tr_e]
        tr_fr = [v / max(1, num_batches) for v in tr_fr]
        tr_reg /= max(1, num_batches)

        # --- Validation ---
        has_dev = dev_loader is not None and len(dev_loader) > 0
        dv_e = [0.0 for _ in range(L)]
        dv_fr = [0.0 for _ in range(L)]
        dv_reg = 0.0

        if has_dev:
            assert dev_loader is not None
            for feats, _labels, mask in dev_loader:
                out = model.forward_batch(feats, mask=mask, train=False, return_spikes=False)
                for k in range(L):
                    dv_e[k] += out[f"loss_e{k}"].item()
                    dv_fr[k] += out[f"fr_l{k + 1}_hz"].item()
                dv_reg += out["loss_reg"].item()
            dv_e = [v / len(dev_loader) for v in dv_e]
            dv_fr = [v / len(dev_loader) for v in dv_fr]
            dv_reg /= len(dev_loader)

        ep_sec = time.time() - t_ep

        # --- History 記録 ---
        history["epoch"].append(float(epoch))
        history["lr"].append(float(current_lr))
        history["train_reg"].append(tr_reg)
        history["epoch_sec"].append(ep_sec)
        for k in range(L):
            history[f"train_e{k}"].append(tr_e[k])
            history[f"train_fr_l{k + 1}"].append(tr_fr[k])
        if has_dev:
            history["dev_reg"].append(dv_reg)
            for k in range(L):
                history[f"dev_e{k}"].append(dv_e[k])
                history[f"dev_fr_l{k + 1}"].append(dv_fr[k])

        # --- チェックポイント保存 ---
        monitor_metric = dv_e[0] if has_dev else tr_e[0]
        is_best = monitor_metric < best_metric
        if is_best:
            best_metric = monitor_metric
            save_checkpoint(os.path.join(cfg.save_dir, "best_model.pt"), model, epoch, best_metric, cfg)

        save_checkpoint(os.path.join(cfg.save_dir, "last_model.pt"), model, epoch, best_metric, cfg)
        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(
                os.path.join(cfg.save_dir, f"epoch_{epoch:02d}.pt"), model, epoch, best_metric, cfg
            )

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        save_training_curves(
            history, L, cfg.f_target_hz, os.path.join(cfg.save_dir, "training_curves.png")
        )

        # --- エポックサマリ表示 ---
        e_sum_str = ", ".join(f"E{k}:{tr_e[k]:.4f}" for k in range(L))
        fr_sum_str = ", ".join(f"L{k + 1}:{tr_fr[k]:.1f}" for k in range(L))
        msg = (
            f"Epoch {epoch:02d}/{cfg.epochs:02d} ({ep_sec / 60.0:.1f} min) | "
            f"Train {e_sum_str}, Reg:{tr_reg:.5f}, FR[Hz]({fr_sum_str})"
        )
        if has_dev:
            dv_str = ", ".join(f"E{k}:{dv_e[k]:.4f}" for k in range(L))
            msg += f" | Dev {dv_str}"
        if is_best:
            msg += " [BEST]"
        print(msg)

    print(f"\nTraining finished. Best model saved to: {os.path.join(cfg.save_dir, 'best_model.pt')}")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-layer HP E-prop on TIMIT.")
    parser.add_argument("--timit_dir", type=str, default="/home/keitaro-sunagawa/Master-Research/TIMIT")
    parser.add_argument("--feature_type", type=str, default="fbank", choices=["mfcc", "fbank"])
    parser.add_argument("--n_layers", type=int, default=4, help="Number of cortical layers (e.g., 2, 3, 4, 6)")
    parser.add_argument("--n_hidden", type=int, default=256, help="Neurons per layer (if --layer_sizes is not set)")
    parser.add_argument(
        "--layer_sizes",
        type=str,
        default="",
        help="Optional comma-separated sizes per layer, e.g. '256,128,128,128'",
    )
    parser.add_argument("--rec_sparsity", type=float, default=0.9)
    parser.add_argument("--frame_repeat_ms", type=float, default=5.0)
    parser.add_argument(
        "--update_interval_ms",
        type=float,
        default=10.0,
        help="Online weight update interval [ms] (<=0 for end-of-batch)",
    )
    parser.add_argument("--refractory_ms", type=float, default=5.0)
    parser.add_argument("--tau_mem_min", type=float, default=10.0)
    parser.add_argument("--tau_mem_max", type=float, default=40.0)
    parser.add_argument("--tau_alif_min", type=float, default=100.0)
    parser.add_argument("--tau_alif_max", type=float, default=300.0)
    parser.add_argument("--tau_r", type=float, default=2.0)
    parser.add_argument("--tau_d", type=float, default=5.0)
    parser.add_argument("--c_reg", type=float, default=20.0)
    parser.add_argument("--f_target_hz", type=float, default=10.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_decay_every", type=int, default=0, help="Decay LR every N epochs (0 = disabled)")
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--save_dir", type=str, default="./checkpoints/hp_eprop")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    parsed_layer_sizes: Optional[Tuple[int, ...]] = None
    if args.layer_sizes.strip():
        parsed_layer_sizes = tuple(int(x.strip()) for x in args.layer_sizes.split(",") if x.strip())
        n_layers = len(parsed_layer_sizes)
    else:
        n_layers = args.n_layers

    config = HPEPropConfig(
        processed_timit_dir=args.timit_dir,
        feature_type=args.feature_type,
        n_layers=n_layers,
        n_hidden=args.n_hidden,
        layer_sizes=parsed_layer_sizes,
        rec_sparsity=args.rec_sparsity,
        frame_repeat_ms=args.frame_repeat_ms,
        update_interval_ms=args.update_interval_ms,
        refractory_ms=args.refractory_ms,
        tau_mem_min=args.tau_mem_min,
        tau_mem_max=args.tau_mem_max,
        tau_alif_min=args.tau_alif_min,
        tau_alif_max=args.tau_alif_max,
        tau_r=args.tau_r,
        tau_d=args.tau_d,
        c_reg=args.c_reg,
        f_target_hz=args.f_target_hz,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lr_decay_every=args.lr_decay_every,
        lr_decay=args.lr_decay,
        save_dir=args.save_dir,
        save_every=args.save_every,
        resume=args.resume,
        seed=args.seed,
    )
    train_timit(config)