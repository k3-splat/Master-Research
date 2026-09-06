# Network.py
#
# Predictive E-prop recurrent spiking neural network
#
# 必要ライブラリ:
#   numpy
#
# 実装内容:
#   - LIF / ALIF ニューロン
#   - 予測誤差入力
#   - 自己フィードバック
#   - 二重指数型スパイクフィルタ
#   - e-prop による再帰結合の学習
#   - 出力結合の勾配降下
#   - ホームオスタシス
#   - Ornstein-Uhlenbeck ノイズ

import numpy as np


class PredictiveEpropNetwork:
    """
    Predictive E-prop RSNN.

    入力:
        prediction_error = prediction - target

    出力:
        y = ネットワークによる予測

    学習:
        W_rec : e-prop
        W_out : 誤差 x filtered spike による勾配降下
        W_in, W_fb : 固定
    """

    def __init__(
        self,
        n_lif=100,
        n_alif=200,
        input_dim=1,
        output_dim=1,
        dt_ms=1.0,
        input_gain=150.0,
        recurrent_sparsity=0.99,
        eta_rec=4e-4,
        eta_out=4e-4,
        weight_decay=5e-6,
        homeo_strength=2.0,
        target_rate_hz=10.0,
        homeo_interval_ms=10,
        tau_mem_ms=20.0,
        tau_alif_ms=2000.0,
        tau_r_ms=50.0,
        tau_d_ms=125.0,
        threshold=0.6,
        bias_current=0.02,
        ou_tau_ms=250.0,
        ou_sigma=1.0,
        ou_mu=0.0,
        surrogate_gamma=0.3,
        refractory_ms=5.0,
        seed=0,
    ):
        self.rng = np.random.default_rng(seed)

        self.n_lif = n_lif
        self.n_alif = n_alif
        self.n = n_lif + n_alif

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dt_ms = dt_ms
        self.dt = dt_ms / 1000.0

        self.eta_rec = eta_rec
        self.eta_out = eta_out
        self.weight_decay = weight_decay

        self.homeo_strength = homeo_strength
        self.target_rate_hz = target_rate_hz
        self.homeo_interval = max(
            1, int(round(homeo_interval_ms / dt_ms))
        )

        self.tau_mem_ms = tau_mem_ms
        self.tau_alif_ms = tau_alif_ms
        self.tau_r_ms = tau_r_ms
        self.tau_d_ms = tau_d_ms

        self.alpha = np.exp(-dt_ms / tau_mem_ms)
        self.rho = np.exp(-dt_ms / tau_alif_ms)

        self.threshold = threshold
        self.bias_current = bias_current

        self.ou_tau_ms = ou_tau_ms
        self.ou_sigma = ou_sigma
        self.ou_mu = ou_mu
        self.ou_alpha = np.exp(-dt_ms / ou_tau_ms)

        self.surrogate_gamma = surrogate_gamma

        self.refractory_steps = max(
            1, int(round(refractory_ms / dt_ms))
        )

        # ------------------------------------------------------------
        # 固定結合
        # ------------------------------------------------------------

        # 論文では W_in, W_fb は N(0,1)/sqrt(g) として初期化される。
        self.W_in = (
            self.rng.normal(
                0.0, 1.0 / np.sqrt(input_gain),
                size=(self.n, input_dim)
            )
        )

        self.W_fb = (
            self.rng.normal(
                0.0, 1.0 / np.sqrt(input_gain),
                size=(self.n, output_dim)
            )
        )

        # ------------------------------------------------------------
        # 学習結合
        # ------------------------------------------------------------

        std_rec = 1.0 / np.sqrt(self.n)
        self.W_rec = self.rng.normal(
            0.0, std_rec, size=(self.n, self.n)
        )

        std_out = 1.0 / np.sqrt(self.n)
        self.W_out = self.rng.normal(
            0.0, std_out, size=(output_dim, self.n)
        )

        # 99% sparsity -> 約1%だけを使用
        self.recurrent_mask = (
            self.rng.random((self.n, self.n))
            > recurrent_sparsity
        ).astype(np.float32)

        # 自己結合を除去
        np.fill_diagonal(self.recurrent_mask, 0.0)

        self.W_rec *= self.recurrent_mask

        # e-prop用のbroadcast alignment行列
        # 出力誤差を各再帰ニューロンへ伝える
        self.B = self.rng.normal(
            0.0,
            1.0 / np.sqrt(output_dim),
            size=(self.n, output_dim)
        )

        # ------------------------------------------------------------
        # ニューロン状態
        # ------------------------------------------------------------

        self.v = np.zeros(self.n)
        self.a = np.zeros(self.n)
        self.z = np.zeros(self.n)

        # 不応期カウンタ
        self.refractory = np.zeros(self.n, dtype=np.int32)

        # スパイクフィルタの状態
        self.q = np.zeros(self.n)
        self.z_bar = np.zeros(self.n)

        # 膜電位に加えるOUノイズ
        self.ou_noise = np.zeros(self.n)

        # 直前の出力
        self.y = np.zeros(output_dim)

        # ------------------------------------------------------------
        # e-prop eligibility trace
        # ------------------------------------------------------------

        # presynaptic filtered spike trace
        self.pre_trace = np.zeros(self.n)

        # ALIF閾値状態に由来するeligibility
        self.eps_b = np.zeros((self.n, self.n))

        # ------------------------------------------------------------
        # ホームオスタシス
        # ------------------------------------------------------------

        self.spike_counter = np.zeros(self.n)
        self.steps_since_homeostasis = 0

        # ALIF用係数
        self.beta = np.zeros(self.n)
        self.beta[n_lif:] = 0.07

    # -----------------------------------------------------------------
    # 状態リセット
    # -----------------------------------------------------------------

    def reset_state(self):
        """
        ニューロン状態をリセットする。
        重みは保持される。
        """
        self.v.fill(0.0)
        self.a.fill(0.0)
        self.z.fill(0.0)
        self.refractory.fill(0)

        self.q.fill(0.0)
        self.z_bar.fill(0.0)
        self.ou_noise.fill(0.0)

        self.y.fill(0.0)

        self.pre_trace.fill(0.0)
        self.eps_b.fill(0.0)

        self.spike_counter.fill(0.0)
        self.steps_since_homeostasis = 0

    # -----------------------------------------------------------------
    # Ornstein-Uhlenbeck ノイズ
    # -----------------------------------------------------------------

    def _update_ou_noise(self):
        """
        時間相関を持つOrnstein-Uhlenbeckノイズ。
        """
        random_term = self.rng.normal(size=self.n)

        self.ou_noise = (
            self.ou_alpha * self.ou_noise
            + (1.0 - self.ou_alpha) * self.ou_mu
            + self.ou_sigma
            * np.sqrt(max(1e-12, 1.0 - self.ou_alpha ** 2))
            * random_term
        )

    # -----------------------------------------------------------------
    # surrogate gradient
    # -----------------------------------------------------------------

    def _surrogate_derivative(self, v, threshold):
        """
        論文で使われているtent型surrogate gradient。
        """
        value = 1.0 - np.abs((v - threshold) / self.threshold)
        value = np.maximum(0.0, value)

        return self.surrogate_gamma * value / self.threshold

    # -----------------------------------------------------------------
    # 1 timestep進める
    # -----------------------------------------------------------------

    def step(
        self,
        prediction_error,
        target=None,
        learn=False,
        error_input_enabled=True,
    ):
        """
        ネットワークを1 timestep進める。

        Parameters
        ----------
        prediction_error : ndarray, shape=(input_dim,)
            y - x

        target : ndarray or None
            学習対象 x。
            learn=True の場合に必要。

        learn : bool
            e-propおよび出力層を更新するか。

        error_input_enabled : bool
            True ならprediction_errorをネットワークへ入力。
            free-running phaseではFalseにする。

        Returns
        -------
        y : ndarray, shape=(output_dim,)
            ネットワーク出力
        z : ndarray, shape=(n,)
            現時刻のスパイク
        """

        prediction_error = np.asarray(
            prediction_error, dtype=np.float64
        ).reshape(self.input_dim)

        if target is not None:
            target = np.asarray(target, dtype=np.float64)
            target = target.reshape(self.output_dim)

        # ------------------------------------------------------------
        # 入力
        # ------------------------------------------------------------

        if error_input_enabled:
            external_input = self.W_in @ prediction_error
        else:
            external_input = np.zeros(self.n)

        recurrent_input = self.W_rec @ self.z_bar
        feedback_input = self.W_fb @ self.y

        # ------------------------------------------------------------
        # 膜電位更新
        # ------------------------------------------------------------

        self._update_ou_noise()

        total_input = (
            external_input
            + recurrent_input
            + feedback_input
            + self.bias_current
            + self.ou_noise
        )

        old_v = self.v.copy()

        self.v = self.alpha * self.v + total_input

        # ALIF閾値
        thresholds = self.threshold + self.beta * self.a

        # 不応期中は膜電位を固定
        refractory_mask = self.refractory > 0
        self.v[refractory_mask] = old_v[refractory_mask]

        # 発火判定
        self.z = (self.v >= thresholds).astype(np.float64)

        # 不応期更新
        self.refractory[self.refractory > 0] -= 1
        self.refractory[self.z > 0] = self.refractory_steps

        # ALIFの適応閾値状態
        self.a = self.rho * self.a + self.z

        # 発火後の再分極
        self.v[self.z > 0] -= thresholds[self.z > 0]

        # ------------------------------------------------------------
        # 二重指数スパイクフィルタ
        # ------------------------------------------------------------

        decay_d = np.exp(-self.dt_ms / self.tau_d_ms)
        decay_r = np.exp(-self.dt_ms / self.tau_r_ms)

        self.z_bar = decay_d * self.z_bar + self.q
        self.q = (
            decay_r * self.q
            + self.z / (self.tau_r_ms * self.tau_d_ms)
        )

        # 現在のfiltered spikeから出力を生成
        self.y = self.W_out @ self.z_bar

        # ------------------------------------------------------------
        # e-prop eligibility trace
        # ------------------------------------------------------------

        # 前シナプス側の活動履歴
        self.pre_trace = (
            self.alpha * self.pre_trace
            + self.z_bar
        )

        psi = self._surrogate_derivative(self.v, thresholds)

        # ALIF閾値状態に関するeligibility
        shifted_pre = np.roll(self.z_bar, 1)

        self.eps_b = (
            psi[:, None] * shifted_pre[None, :]
            + (
                self.rho
                - self.beta[:, None] * psi[:, None]
            ) * self.eps_b
        )

        # e_ij
        eligibility = psi[:, None] * (
            self.pre_trace[None, :]
            - self.beta[:, None] * self.eps_b
        )

        # 疎結合を維持
        eligibility *= self.recurrent_mask

        # 発火率を記録
        self.spike_counter += self.z
        self.steps_since_homeostasis += 1

        # ------------------------------------------------------------
        # 学習
        # ------------------------------------------------------------

        if learn:
            if target is None:
                raise ValueError(
                    "learn=True の場合は target が必要です。"
                )

            # 予測誤差
            output_error = self.y - target

            # --------------------------------------------------------
            # 出力結合
            # --------------------------------------------------------

            grad_out = np.outer(
                output_error,
                self.z_bar
            )

            self.W_out -= self.eta_out * grad_out

            # --------------------------------------------------------
            # 再帰結合：e-prop
            # --------------------------------------------------------

            # prediction errorをbroadcast alignmentする
            learning_signal = self.B @ output_error

            # e-prop update
            grad_rec = learning_signal[:, None] * eligibility

            self.W_rec -= self.eta_rec * grad_rec

            # 疎結合を維持
            self.W_rec *= self.recurrent_mask

            # weight decay
            self.W_rec *= (1.0 - self.eta_rec * self.weight_decay)
            self.W_out *= (1.0 - self.eta_out * self.weight_decay)

        # ------------------------------------------------------------
        # ホームオスタシス
        # ------------------------------------------------------------

        if self.steps_since_homeostasis >= self.homeo_interval:
            self._apply_homeostasis()

        return self.y.copy(), self.z.copy()

    # -----------------------------------------------------------------
    # ホームオスタシス
    # -----------------------------------------------------------------

    def _apply_homeostasis(self):
        """
        平均発火率をtarget_rate_hzへ近づける。

        論文の発火率正則化に対応する近似実装である。
        """
        interval_seconds = self.steps_since_homeostasis * self.dt

        firing_rate = (
            self.spike_counter / max(interval_seconds, 1e-12)
        )

        rate_error = firing_rate - self.target_rate_hz

        # 高発火ニューロンの再帰入力を弱め、
        # 低発火ニューロンの再帰入力を相対的に強める。
        row_scale = np.exp(
            -self.homeo_strength
            * 1e-4
            * rate_error
        )

        row_scale = np.clip(row_scale, 0.95, 1.05)

        self.W_rec *= row_scale[:, None]
        self.W_rec *= self.recurrent_mask

        self.spike_counter.fill(0.0)
        self.steps_since_homeostasis = 0

    # -----------------------------------------------------------------
    # 1 phaseをまとめて実行
    # -----------------------------------------------------------------

    def run_phase(
        self,
        targets,
        learn=False,
        error_input_enabled=True,
        reset_state=True,
    ):
        """
        targetsを時系列としてphaseを実行する。

        Parameters
        ----------
        targets : ndarray, shape=(time, output_dim)
            目標時系列

        learn : bool
            学習するか

        error_input_enabled : bool
            prediction errorを入力するか

        reset_state : bool
            phase開始時に内部状態をリセットするか
        """

        targets = np.asarray(targets, dtype=np.float64)

        if targets.ndim == 1:
            targets = targets[:, None]

        if reset_state:
            self.reset_state()

        outputs = []
        spikes = []

        for target in targets:
            # 現時刻の予測誤差
            error = self.y - target

            y, z = self.step(
                prediction_error=error,
                target=target,
                learn=learn,
                error_input_enabled=error_input_enabled,
            )

            outputs.append(y)
            spikes.append(z)

        return np.asarray(outputs), np.asarray(spikes)
