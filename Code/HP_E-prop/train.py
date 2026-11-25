#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical Predictive E-prop (HP E-prop) with TIMIT Dataset Pipeline.
Requires `encoder.py` in the same directory or Python path.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from encoder import TimitProcessedDataset, timit_collate_fn


@dataclass
class HPEPropConfig:
    """HP E-prop (Hierarchical Predictive E-prop) のハイパーパラメータ管理用 dataclass"""

    # データセット設定 (encoder.py の出力形式に対応)
    processed_timit_dir: str = "/home/keitaro-sunagawa/Master-Research/TIMIT/"
    feature_type: str = "fbank"  # "mfcc" (39次元) または "fbank" (123次元)

    # ネットワーク構造 (入力次元, 第1層 A1, 第2層 STG)
    n_in: int = 39              # __post_init__ で feature_type に合わせて自動設定
    n_l1: int = 128             # 第1層 (一次聴覚野 A1 相当)
    n_l2: int = 128             # 第2層 (高次聴覚野 STG 相当)
    alif_ratio_l1: float = 0.5  # 第1層の ALIF ニューロン比率
    alif_ratio_l2: float = 0.8  # 第2層の ALIF ニューロン比率

    # 時間・同一入力・不応期設定 [ms]
    dt: float = 1.0
    frame_repeat_ms: float = 5.0  # E-prop にならい 5ms 同一入力を継続
    refractory_ms: float = 5.0    # 不応期の長さ [ms] (Predictive E-prop にならい 5ms)

    # ニューロン膜時定数・適応閾値時定数 [ms] (tau_mem_l1 < tau_mem_l2 で時間階層性を形成)
    tau_mem_l1: float = 10.0
    tau_mem_l2: float = 40.0
    tau_alif_l1: float = 100.0
    tau_alif_l2: float = 300.0

    # 閾値・適応強度・バイアス・疑似微分パラメータ
    v_th: float = 0.6
    beta_l1: float = 0.07
    beta_l2: float = 0.07
    i_bias_l1: float = 0.02
    i_bias_l2: float = 0.02
    gamma_d: float = 0.3

    # シナプス平滑化用 因果的ダブル指数フィルタ時定数 [ms]
    tau_r: float = 20.0
    tau_d: float = 50.0

    # 発火率正則化 (E-prop Supplementary Note 2)
    c_reg: float = 1.0          # 正則化強度係数 (0.0 で無効化)
    f_target_hz: float = 10.0   # 目標発火率 [Hz]

    # 学習・最適化パラメータ (バッチごとに重み更新)
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    use_adam: bool = True
    seed: int = 42

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

    @property
    def frame_repeat_steps(self) -> int:
        return max(1, int(round(self.frame_repeat_ms / self.dt)))

    @property
    def refractory_steps(self) -> int:
        return max(0, int(round(self.refractory_ms / self.dt)))

    @property
    def f_target_per_step(self) -> float:
        return self.f_target_hz * (self.dt / 1000.0)

    @property
    def alpha_l1(self) -> float:
        return math.exp(-self.dt / self.tau_mem_l1)

    @property
    def alpha_l2(self) -> float:
        return math.exp(-self.dt / self.tau_mem_l2)

    @property
    def rho_l1(self) -> float:
        return math.exp(-self.dt / self.tau_alif_l1)

    @property
    def rho_l2(self) -> float:
        return math.exp(-self.dt / self.tau_alif_l2)

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
        # 発火率正則化用: バッチ内の各サンプルごとの有効時間ステップ和 \sum_t m_{b,t} e_{b,ji}^t
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
        """
        論文 式 (1), (34)-(41), (50)-(51) に基づく適格性トレースと二重フィルタの更新。
        """
        # 1ステップ前の e_bar を退避 (オンライン学習シグナルの1ステップ遅延整合用)
        self.e_bar_prev.copy_(self.e_bar)

        # (B, n_post, 1) にブロードキャスト
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

        # 適格性ベクトルの更新 (式 38)
        new_eps_v = j_vv * self.eps_v + j_va * self.eps_a + pre_u
        new_eps_a = j_av * self.eps_v + j_aa * self.eps_a
        self.eps_v.copy_(new_eps_v)
        self.eps_a.copy_(new_eps_a)

        # 適格性トレース e^t (式 41)
        e_curr = psi_c * (self.eps_v - beta_u * self.eps_a)

        # 発火率正則化用にフィルタ前の e^t を各サンプルごとにマスク付きで累積
        if mask_t is not None:
            self.e_sum.add_(e_curr * mask_t.unsqueeze(-1))
        else:
            self.e_sum.add_(e_curr)

        # ダブル指数フィルタによる平滑化 (式 50, 51)
        self.e_bar.mul_(self.decay_d).add_(self.q_e)
        self.q_e.mul_(self.decay_r).add_(e_curr, alpha=self.syn_scale)

        return self.e_bar


class HPEPropLayer:
    """
    HP E-prop の単一皮質層 (ALIF + LIF 混在ニューロン群) の状態と動態を管理するクラス。
    """

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

        # ALIF と LIF のマスク (LIF は beta = 0)
        n_alif = int(round(n_neurons * alif_ratio))
        beta_vec = torch.zeros(1, n_neurons, device=self.device, dtype=self.dtype)
        if n_alif > 0:
            beta_vec[:, :n_alif] = beta_val
        self.beta = beta_vec

        # 状態変数 (reset_state で初期化)
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

        # 1ステップ前の状態 (適格性トレース更新用)
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
        """
        現在時刻 t の膜電位 v^t と適応閾値 b^t から、スパイク z^t・疑似微分 psi^t・
        平滑化スパイク z_bar^t を計算する。
        不応期中はスパイクを 0、疑似微分を 0 とする。
        """
        # 履歴のシフト
        self.psi_prev.copy_(self.psi)
        self.b_prev.copy_(self.b)
        self.z_prev.copy_(self.z)
        self.z_bar_prev2.copy_(self.z_bar_prev)
        self.z_bar_prev.copy_(self.z_bar)

        # 適応閾値 b^t = v_th + beta * a^t (式 6)
        self.b = self.v_th + self.beta * self.a

        # 不応期マスク (ref_count > 0)
        in_ref = self.ref_count > 0

        # 発火判定 (式 5): 不応期中は強制的に 0
        fired = (self.v >= self.b) & (~in_ref)
        self.z = fired.to(self.dtype)

        # 疑似微分 psi^t (式 9): 不応期中は強制的に 0
        diff = torch.abs(self.v - self.b) / self.v_th
        psi_raw = (self.gamma_d / self.v_th) * torch.clamp(1.0 - diff, min=0.0)
        self.psi = torch.where(in_ref, torch.zeros_like(psi_raw), psi_raw)

        # シナプス平滑化 z_bar^t (式 7, 8)
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
        Predictive E-prop に従い、発火ステップで式 (1) による減算リセット (-z^t * b^t) を行い、
        その後の不応期中 (ref_count > 0) はリセット後の膜電位に固定する。
        """
        in_ref = self.ref_count > 0
        fired = self.z > 0.5

        # 式 (1) に基づく膜電位更新 (発火ステップでは -z^t * b^t による減算リセットを適用)
        v_next = self.alpha * self.v + bottom_input + rec_input + self.i_bias - self.z * self.b
        if e_self is not None:
            v_next = v_next - e_self

        # すでに不応期中のニューロンはリセット後の電位 (現在の self.v) に固定
        self.v = torch.where(in_ref, self.v, v_next)

        # 不応期カウンターの更新
        ref_dec = torch.clamp(self.ref_count - 1, min=0)
        self.ref_count = torch.where(
            fired,
            torch.full_like(self.ref_count, self.refractory_steps),
            ref_dec,
        )

        # 適応閾値変数の更新 a^{t+1} = rho * a^t + z^t (式 6)
        self.a.mul_(self.rho).add_(self.z)


class HPEPropNetwork(nn.Module):
    """
    2層階層的予測符号化 SNN (HP E-prop) ネットワーク本体。
    - 第1層 (l=1): 一次聴覚野 A1 相当
    - 第2層 (l=2): 高次聴覚野 STG 相当
    局所オンライン学習則 (e-prop) + 発火率正則化により W_rec, W_bottom, W_top をバッチごとに更新する。
    """

    def __init__(self, cfg: HPEPropConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dev, dt = cfg.device, cfg.dtype

        # 皮質層の構築
        self.l1 = HPEPropLayer(
            n_neurons=cfg.n_l1,
            alif_ratio=cfg.alif_ratio_l1,
            alpha=cfg.alpha_l1,
            rho=cfg.rho_l1,
            beta_val=cfg.beta_l1,
            i_bias=cfg.i_bias_l1,
            cfg=cfg,
        )
        self.l2 = HPEPropLayer(
            n_neurons=cfg.n_l2,
            alif_ratio=cfg.alif_ratio_l2,
            alpha=cfg.alpha_l2,
            rho=cfg.rho_l2,
            beta_val=cfg.beta_l2,
            i_bias=cfg.i_bias_l2,
            cfg=cfg,
        )

        # 学習対象の重み行列 (形状: [n_post, n_pre])
        self.w_rec_1 = nn.Parameter(self._init_rec_weight(cfg.n_l1, dev, dt))
        self.w_rec_2 = nn.Parameter(self._init_rec_weight(cfg.n_l2, dev, dt))

        self.w_bot_0 = nn.Parameter(torch.randn(cfg.n_l1, cfg.n_in, device=dev, dtype=dt) / math.sqrt(cfg.n_in))
        self.w_bot_1 = nn.Parameter(torch.randn(cfg.n_l2, cfg.n_l1, device=dev, dtype=dt) / math.sqrt(cfg.n_l1))

        self.w_top_0 = nn.Parameter(torch.randn(cfg.n_in, cfg.n_l1, device=dev, dtype=dt) / math.sqrt(cfg.n_l1))
        self.w_top_1 = nn.Parameter(torch.randn(cfg.n_l1, cfg.n_l2, device=dev, dtype=dt) / math.sqrt(cfg.n_l2))

        # 対角成分を0にするマスク (式 4: j != i)
        self.register_buffer("diag_mask_1", 1.0 - torch.eye(cfg.n_l1, device=dev, dtype=dt))
        self.register_buffer("diag_mask_2", 1.0 - torch.eye(cfg.n_l2, device=dev, dtype=dt))

        # 誤差フィードバック用 固定ランダム行列 B^{(l)} (Broadcast Alignment, 式 47)
        self.register_buffer("B_1", torch.randn(cfg.n_l1, cfg.n_in, device=dev, dtype=dt) / math.sqrt(cfg.n_in))
        self.register_buffer("B_2", torch.randn(cfg.n_l2, cfg.n_l1, device=dev, dtype=dt) / math.sqrt(cfg.n_l1))

        # オプティマイザ (手動計算した e-prop 勾配を .grad に格納してバッチごとに適用)
        if cfg.use_adam:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # 適格性トレース群 (reset_state でバッチサイズに合わせて初期化)
        self.et_rec_1: Optional[EligibilityTrace2D] = None
        self.et_rec_2: Optional[EligibilityTrace2D] = None
        self.et_bot_0: Optional[EligibilityTrace2D] = None
        self.et_bot_1: Optional[EligibilityTrace2D] = None
        self.et_top_1_self: Optional[EligibilityTrace2D] = None   # e_{top, ij}^{(1)}
        self.et_top_0_upper: Optional[EligibilityTrace2D] = None  # 因数分解された \tilde{e}_{top, kj}^{(0)}
        self.et_top_1_upper: Optional[EligibilityTrace2D] = None  # 因数分解された \tilde{e}_{top, kj}^{(1)}

        # 1ステップ前の予測誤差
        self.e0_prev: Optional[torch.Tensor] = None
        self.e1_prev: Optional[torch.Tensor] = None

    @staticmethod
    def _init_rec_weight(n: int, device: str, dtype: torch.dtype) -> torch.Tensor:
        w = torch.randn(n, n, device=device, dtype=dtype) / math.sqrt(n)
        w.fill_diagonal_(0.0)
        return w

    @torch.no_grad()
    def reset_state(self, batch_size: int) -> None:
        cfg = self.cfg
        self.l1.reset_state(batch_size)
        self.l2.reset_state(batch_size)

        self.et_rec_1 = EligibilityTrace2D(batch_size, cfg.n_l1, cfg.n_l1, cfg)
        self.et_rec_2 = EligibilityTrace2D(batch_size, cfg.n_l2, cfg.n_l2, cfg)
        self.et_bot_0 = EligibilityTrace2D(batch_size, cfg.n_l1, cfg.n_in, cfg)
        self.et_bot_1 = EligibilityTrace2D(batch_size, cfg.n_l2, cfg.n_l1, cfg)

        # トップダウン重み用の適格性トレース
        self.et_top_1_self = EligibilityTrace2D(batch_size, cfg.n_l1, cfg.n_l2, cfg)
        # 3階テンソルを O(N^2) に圧縮するための因数分解トレース
        self.et_top_0_upper = EligibilityTrace2D(batch_size, cfg.n_l1, cfg.n_l1, cfg)
        self.et_top_1_upper = EligibilityTrace2D(batch_size, cfg.n_l2, cfg.n_l2, cfg)

        self.e0_prev = torch.zeros(batch_size, cfg.n_in, device=cfg.device, dtype=cfg.dtype)
        self.e1_prev = torch.zeros(batch_size, cfg.n_l1, device=cfg.device, dtype=cfg.dtype)

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
        を受け取り、各フレームを 5ms (frame_repeat_steps) 連続入力して
        オンライン順伝播およびバッチごとの重み更新（サンプル別発火率正則化込み）を実行する。
        """
        repeat_steps = self.cfg.frame_repeat_steps
        x_seq = features.repeat_interleave(repeat_steps, dim=1)  # (B, T_frames * 5, n_in)
        batch_size, seq_len, _ = x_seq.shape

        if mask is not None:
            m_seq = mask.repeat_interleave(repeat_steps, dim=1)  # (B, T_frames * 5)
        else:
            m_seq = torch.ones((batch_size, seq_len), device=x_seq.device, dtype=x_seq.dtype)

        # 各サンプルごとの有効時間ステップ数 T_b (B, 1) とバッチ全体の有効ステップ総数
        valid_steps_per_sample = m_seq.sum(dim=1, keepdim=True).clamp(min=1.0)
        valid_steps = m_seq.sum().clamp(min=1.0).item()

        self.reset_state(batch_size)

        if train:
            grad_acc = {name: torch.zeros_like(p) for name, p in self.named_parameters()}
        else:
            grad_acc = {}

        total_loss_e0 = 0.0
        total_loss_e1 = 0.0
        # 発火率正則化用: サンプルごと・ニューロンごとの有効スパイク総数 (B, n_l1), (B, n_l2)
        spike_count_l1 = torch.zeros((batch_size, self.cfg.n_l1), device=x_seq.device, dtype=x_seq.dtype)
        spike_count_l2 = torch.zeros((batch_size, self.cfg.n_l2), device=x_seq.device, dtype=x_seq.dtype)

        spikes_l1 = (
            torch.empty((batch_size, seq_len, self.cfg.n_l1), device=x_seq.device, dtype=x_seq.dtype)
            if return_spikes
            else None
        )
        spikes_l2 = (
            torch.empty((batch_size, seq_len, self.cfg.n_l2), device=x_seq.device, dtype=x_seq.dtype)
            if return_spikes
            else None
        )

        # 対角成分をマスクした再帰重み
        w_rec_1_eff = self.w_rec_1 * self.diag_mask_1
        w_rec_2_eff = self.w_rec_2 * self.diag_mask_2

        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            m_t = m_seq[:, t : t + 1]  # (B, 1)

            # 1. 現在時刻 t の発火・疑似微分・平滑化スパイク計算
            z1_t, z1_bar_t = self.l1.compute_spike_and_surrogate()
            z2_t, z2_bar_t = self.l2.compute_spike_and_surrogate()

            spike_count_l1.add_(z1_t * m_t)
            spike_count_l2.add_(z2_t * m_t)
            if spikes_l1 is not None and spikes_l2 is not None:
                spikes_l1[:, t, :] = z1_t
                spikes_l2[:, t, :] = z2_t

            # 2. トップダウン予測誤差の計算 (式 2, 3) + パディングマスク適用
            pred_0 = torch.matmul(self.l1.z_bar_prev, self.w_top_0.t())
            e0_t = (x_t - pred_0) * m_t

            pred_1 = torch.matmul(self.l2.z_bar_prev, self.w_top_1.t())
            e1_t = (z1_bar_t - pred_1) * m_t

            total_loss_e0 += 0.5 * e0_t.pow(2).sum().item()
            total_loss_e1 += 0.5 * e1_t.pow(2).sum().item()

            # 3. 適格性トレースの更新 & バッチ内オンライン勾配蓄積
            if train:
                assert (
                    self.et_rec_1 is not None
                    and self.et_rec_2 is not None
                    and self.et_bot_0 is not None
                    and self.et_bot_1 is not None
                    and self.et_top_1_self is not None
                    and self.et_top_0_upper is not None
                    and self.et_top_1_upper is not None
                    and self.e0_prev is not None
                    and self.e1_prev is not None
                )

                # --- 適格性トレースの更新 (時刻 t) ---
                self.et_rec_1.update(
                    self.l1.alpha, self.l1.rho, self.l1.beta,
                    self.l1.psi_prev, self.l1.psi, self.l1.b_prev, self.l1.z_prev,
                    self.l1.z_bar_prev, mask_t=m_t,
                )
                self.et_rec_2.update(
                    self.l2.alpha, self.l2.rho, self.l2.beta,
                    self.l2.psi_prev, self.l2.psi, self.l2.b_prev, self.l2.z_prev,
                    self.l2.z_bar_prev, mask_t=m_t,
                )
                self.et_bot_0.update(
                    self.l1.alpha, self.l1.rho, self.l1.beta,
                    self.l1.psi_prev, self.l1.psi, self.l1.b_prev, self.l1.z_prev,
                    self.e0_prev, mask_t=m_t,
                )
                self.et_bot_1.update(
                    self.l2.alpha, self.l2.rho, self.l2.beta,
                    self.l2.psi_prev, self.l2.psi, self.l2.b_prev, self.l2.z_prev,
                    self.e1_prev, mask_t=m_t,
                )
                # W^{top,(1)} の自層経路 e_{top, ij}^{(1)} (式 94, 98)
                self.et_top_1_self.update(
                    self.l1.alpha, self.l1.rho, self.l1.beta,
                    self.l1.psi_prev, self.l1.psi, self.l1.b_prev, self.l1.z_prev,
                    self.l2.z_bar_prev2, mask_t=m_t,
                )
                # W^{top,(0)}, W^{top,(1)} の上位層経路 (因数分解版, 式 103, 105)
                self.et_top_0_upper.update(
                    self.l1.alpha, self.l1.rho, self.l1.beta,
                    self.l1.psi_prev, self.l1.psi, self.l1.b_prev, self.l1.z_prev,
                    self.l1.z_bar_prev2, mask_t=m_t,
                )
                self.et_top_1_upper.update(
                    self.l2.alpha, self.l2.rho, self.l2.beta,
                    self.l2.psi_prev, self.l2.psi, self.l2.b_prev, self.l2.z_prev,
                    self.l2.z_bar_prev2, mask_t=m_t,
                )

                # --- 勾配のオンライン蓄積 (時刻 t-1 の学習シグナルが時刻 t の誤差で確定) ---
                if t > 0:
                    # 学習シグナル L^{(1), t-1} = e^{(1), t-1} - B^{(1)} e^{(0), t} (式 46, 47)
                    L1_prev = self.e1_prev - torch.matmul(e0_t, self.B_1.t())
                    # 学習シグナル L^{(2), t-1} = - B^{(2)} e^{(1), t} (最上位層のため e^{(2)} = 0)
                    L2_prev = -torch.matmul(e1_t, self.B_2.t())

                    # 再帰重み勾配 (式 49)
                    grad_acc["w_rec_1"].add_(
                        torch.einsum("bi,bij->ij", L1_prev, self.et_rec_1.e_bar_prev)
                    )
                    grad_acc["w_rec_2"].add_(
                        torch.einsum("bi,bij->ij", L2_prev, self.et_rec_2.e_bar_prev)
                    )

                    # ボトムアップ重み勾配 (式 53)
                    grad_acc["w_bot_0"].add_(
                        torch.einsum("bi,bij->ij", L1_prev, self.et_bot_0.e_bar_prev)
                    )
                    grad_acc["w_bot_1"].add_(
                        torch.einsum("bi,bij->ij", L2_prev, self.et_bot_1.e_bar_prev)
                    )

                    # トップダウン重み W^{top,(0)} の上位層経路 (因数分解)
                    weighted_top0_upper = torch.einsum(
                        "bk,bkj->kj", L1_prev, self.et_top_0_upper.e_bar_prev
                    )
                    grad_acc["w_top_0"].sub_(torch.matmul(self.w_bot_0.t(), weighted_top0_upper))

                    # トップダウン重み W^{top,(1)} の自層経路 + 上位層経路 (式 108, 109)
                    grad_acc["w_top_1"].add_(
                        torch.einsum("bi,bij->ij", L1_prev, self.et_top_1_self.e_bar_prev)
                    )
                    weighted_top1_upper = torch.einsum(
                        "bk,bkj->kj", L2_prev, self.et_top_1_upper.e_bar_prev
                    )
                    grad_acc["w_top_1"].sub_(torch.matmul(self.w_bot_1.t(), weighted_top1_upper))

                # トップダウン重みの直接経路: - e_i^{(l), t} * \bar{z}_j^{(l+1), t-1} (式 109)
                grad_acc["w_top_0"].sub_(torch.einsum("bi,bj->ij", e0_t, self.l1.z_bar_prev))
                grad_acc["w_top_1"].sub_(torch.einsum("bi,bj->ij", e1_t, self.l2.z_bar_prev))

            # 4. 層内再帰入力とボトムアップ入力の計算 (式 1, 4)
            i_rec_1 = torch.matmul(z1_bar_t, w_rec_1_eff.t())
            i_rec_2 = torch.matmul(z2_bar_t, w_rec_2_eff.t())
            i_bot_1 = torch.matmul(e0_t, self.w_bot_0.t())
            i_bot_2 = torch.matmul(e1_t, self.w_bot_1.t())

            # 5. 次時刻 t+1 への膜電位・不応期・適応閾値の更新
            self.l1.update_membrane_and_adaptation(e_self=e1_t, bottom_input=i_bot_1, rec_input=i_rec_1)
            self.l2.update_membrane_and_adaptation(e_self=None, bottom_input=i_bot_2, rec_input=i_rec_2)

            # 誤差の履歴更新
            self.e0_prev = e0_t
            self.e1_prev = e1_t

        # 各サンプルごと・ニューロンごとの平均発火率 f_{b,j}^{av} [spikes / step] (B, n_l)
        f_av_l1 = spike_count_l1 / valid_steps_per_sample
        f_av_l2 = spike_count_l2 / valid_steps_per_sample
        f_target = self.cfg.f_target_per_step

        # 発火率正則化損失 (サンプルごとの損失のバッチ平均)
        loss_reg_l1 = 0.5 * (f_av_l1 - f_target).pow(2).sum(dim=-1).mean().item()
        loss_reg_l2 = 0.5 * (f_av_l2 - f_target).pow(2).sum(dim=-1).mean().item()

        # 6. バッチごとのパラメータ更新 (予測誤差勾配 + サンプル別発火率正則化勾配をバッチサイズで平均化)
        if train:
            num_sequences = mask.size(0) if mask is not None else batch_size

            # サンプルごとの発火率正則化勾配を grad_acc に加算
            if self.cfg.c_reg > 0.0:
                assert (
                    self.et_rec_1 is not None
                    and self.et_rec_2 is not None
                    and self.et_bot_0 is not None
                    and self.et_bot_1 is not None
                    and self.et_top_1_self is not None
                    and self.et_top_0_upper is not None
                    and self.et_top_1_upper is not None
                )
                # サンプルごとの正則化学習シグナル L_{reg, b}^{(l)} = c_reg * (f_{b}^{av} - f^{target}) / T_b  (B, n_l)
                L_reg_1 = self.cfg.c_reg * (f_av_l1 - f_target) / valid_steps_per_sample
                L_reg_2 = self.cfg.c_reg * (f_av_l2 - f_target) / valid_steps_per_sample

                grad_acc["w_rec_1"].add_(torch.einsum("bi,bij->ij", L_reg_1, self.et_rec_1.e_sum))
                grad_acc["w_rec_2"].add_(torch.einsum("bi,bij->ij", L_reg_2, self.et_rec_2.e_sum))
                grad_acc["w_bot_0"].add_(torch.einsum("bi,bij->ij", L_reg_1, self.et_bot_0.e_sum))
                grad_acc["w_bot_1"].add_(torch.einsum("bi,bij->ij", L_reg_2, self.et_bot_1.e_sum))

                weighted_reg_top0 = torch.einsum("bk,bkj->kj", L_reg_1, self.et_top_0_upper.e_sum)
                grad_acc["w_top_0"].sub_(torch.matmul(self.w_bot_0.t(), weighted_reg_top0))

                grad_acc["w_top_1"].add_(torch.einsum("bi,bij->ij", L_reg_1, self.et_top_1_self.e_sum))
                weighted_reg_top1 = torch.einsum("bk,bkj->kj", L_reg_2, self.et_top_1_upper.e_sum)
                grad_acc["w_top_1"].sub_(torch.matmul(self.w_bot_1.t(), weighted_reg_top1))

            self.optimizer.zero_grad()
            for name, p in self.named_parameters():
                g = grad_acc[name] / num_sequences
                if name == "w_rec_1":
                    g = g * self.diag_mask_1
                elif name == "w_rec_2":
                    g = g * self.diag_mask_2
                p.grad = g

            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.grad_clip)

            self.optimizer.step()

            # 対角成分を厳密に 0 に維持
            self.w_rec_1.mul_(self.diag_mask_1)
            self.w_rec_2.mul_(self.diag_mask_2)

        fr_l1_hz = f_av_l1.mean().item() * (1000.0 / self.cfg.dt)
        fr_l2_hz = f_av_l2.mean().item() * (1000.0 / self.cfg.dt)

        result: Dict[str, torch.Tensor] = {
            "loss_e0": torch.tensor(total_loss_e0 / valid_steps),
            "loss_e1": torch.tensor(total_loss_e1 / valid_steps),
            "loss_reg": torch.tensor(loss_reg_l1 + loss_reg_l2),
            "fr_l1_hz": torch.tensor(fr_l1_hz),
            "fr_l2_hz": torch.tensor(fr_l2_hz),
        }
        if spikes_l1 is not None and spikes_l2 is not None:
            result["spikes_l1"] = spikes_l1
            result["spikes_l2"] = spikes_l2
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


def train_timit(cfg: HPEPropConfig) -> HPEPropNetwork:
    """TIMIT データセットを用いた HP E-prop の自己教師あり学習ループ"""
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    print(
        f"Initializing HP E-prop on {cfg.device} | "
        f"feature={cfg.feature_type} (dim={cfg.n_in}) | "
        f"repeat={cfg.frame_repeat_ms}ms ({cfg.frame_repeat_steps} steps) | "
        f"batch_size={cfg.batch_size} | "
        f"c_reg={cfg.c_reg}, f_target={cfg.f_target_hz}Hz"
    )

    train_loader, dev_loader = build_timit_dataloaders(cfg)
    model = HPEPropNetwork(cfg).to(cfg.device)

    for epoch in range(1, cfg.epochs + 1):
        # --- Training (バッチごとに重み更新) ---
        tr_e0, tr_e1, tr_reg, tr_fr1, tr_fr2 = 0.0, 0.0, 0.0, 0.0, 0.0
        num_batches = len(train_loader)

        for batch_idx, (feats, _labels, mask) in enumerate(train_loader, start=1):
            out = model.forward_batch(feats, mask=mask, train=True, return_spikes=False)
            tr_e0 += out["loss_e0"].item()
            tr_e1 += out["loss_e1"].item()
            tr_reg += out["loss_reg"].item()
            tr_fr1 += out["fr_l1_hz"].item()
            tr_fr2 += out["fr_l2_hz"].item()

            if batch_idx % 50 == 0 or batch_idx == num_batches:
                print(
                    f"  Epoch {epoch:02d} [{batch_idx:04d}/{num_batches:04d}] | "
                    f"E0: {out['loss_e0'].item():.4f} | E1: {out['loss_e1'].item():.4f} | "
                    f"Reg: {out['loss_reg'].item():.6f} | "
                    f"FR L1: {out['fr_l1_hz'].item():.1f} Hz | FR L2: {out['fr_l2_hz'].item():.1f} Hz"
                )

        tr_e0 /= max(1, num_batches)
        tr_e1 /= max(1, num_batches)
        tr_reg /= max(1, num_batches)
        tr_fr1 /= max(1, num_batches)
        tr_fr2 /= max(1, num_batches)

        msg = (
            f"Epoch {epoch:02d} Summary | "
            f"Train E0: {tr_e0:.4f}, E1: {tr_e1:.4f}, Reg: {tr_reg:.6f}, "
            f"FR1: {tr_fr1:.1f} Hz, FR2: {tr_fr2:.1f} Hz"
        )

        # --- Validation (develop セットがあれば評価) ---
        if dev_loader is not None and len(dev_loader) > 0:
            dv_e0, dv_e1 = 0.0, 0.0
            for feats, _labels, mask in dev_loader:
                out = model.forward_batch(feats, mask=mask, train=False, return_spikes=False)
                dv_e0 += out["loss_e0"].item()
                dv_e1 += out["loss_e1"].item()
            dv_e0 /= len(dev_loader)
            dv_e1 /= len(dev_loader)
            msg += f" | Dev E0: {dv_e0:.4f}, E1: {dv_e1:.4f}"

        print(msg)

    return model


if __name__ == "__main__":
    config = HPEPropConfig(
        processed_timit_dir="/home/keitaro-sunagawa/Master-Research/TIMIT/",
        feature_type="fbank",
        n_l1=128,
        n_l2=128,
        frame_repeat_ms=5.0,
        refractory_ms=5.0,
        c_reg=1.0,
        f_target_hz=10.0,
        batch_size=8,
        epochs=20,
        lr=1e-3,
    )
    train_timit(config)