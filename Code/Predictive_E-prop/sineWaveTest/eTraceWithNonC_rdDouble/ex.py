import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from dataclasses import dataclass, asdict

# デバイス設定 (CUDA -> MPS -> CPU)
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")


# ==========================================
# ハイパーパラメータ一括管理クラス
# ==========================================
@dataclass
class Config:
    # シミュレーション時間設定 (ms)
    training_phase: int = 10000
    error_driven_phase: int = 5000
    free_running_phase: int = 5000
    time_step_dt: float = 1.0

    # 学習・最適化パラメータ
    eta_rec: float = 0.0004
    eta_out: float = 0.0004
    lambda_w: float = 5e-6
    lambda_reg: float = 2.0
    t_delay: int = 10
    f_star: float = 0.01

    # ターゲット信号パラメータ (正弦波)
    A: float = 0.4
    T: float = 1000.0
    phi: float = 0.0
    c: float = 0.0

    # ネットワーク構造
    n_inputs: int = 1
    n_lif: int = 100
    n_alif: int = 200
    n_outputs: int = 1
    density: float = 0.01

    # ニューロン・シナプスダイナミクス
    gamma_d: float = 0.3
    v_th: float = 0.6
    I_bias: float = 0.02
    tau_mem: float = 20.0
    tau_alif: float = 2000.0
    beta: float = 0.0174
    tau_r: float = 50.0
    tau_d: float = 125.0
    t_ref: float = 5.0
    g: float = 150.0

    # OUノイズパラメータ
    tau_s: float = 250.0
    sigma_s: float = 1.0
    mu_s: float = 0.0

    # 実行設定
    epochs: int = 50

    @property
    def n_neurons(self) -> int:
        return self.n_lif + self.n_alif

    @property
    def total_time(self) -> int:
        return self.training_phase + self.error_driven_phase + self.free_running_phase


# ==========================================
# ニューロンモデル
# ==========================================
class BaseNeuronGroup:
    def __init__(self, n_neurons, cfg: Config, device=device):
        self.n = n_neurons
        self.cfg = cfg
        self.device = device

        self.alpha = float(np.exp(-cfg.time_step_dt / cfg.tau_mem))
        self.decay_r = float(np.exp(-cfg.time_step_dt / cfg.tau_r))
        self.decay_d = float(np.exp(-cfg.time_step_dt / cfg.tau_d))
        self.c_rd = 1.0 / (cfg.tau_r * cfg.tau_d)

        self.reset_state()

    def reset_state(self):
        self.v = torch.zeros(self.n, device=self.device, dtype=torch.float32)
        self.z = torch.zeros(self.n, device=self.device, dtype=torch.float32)
        self.z_bar = torch.zeros(self.n, device=self.device, dtype=torch.float32)
        self.z_bar_bar = torch.zeros(self.n, device=self.device, dtype=torch.float32)

        # 順方向の二重フィルタ状態
        self.q = torch.full((self.n,), self.c_rd, device=self.device, dtype=torch.float32)

        self.psi = torch.zeros(self.n, device=self.device, dtype=torch.float32)
        self.b = torch.full((self.n,), self.cfg.v_th, device=self.device, dtype=torch.float32)
        self.ref_counts = torch.zeros(self.n, device=self.device, dtype=torch.float32)

    def _update_spikes_and_filters(self, v_next_raw, b_next):
        self.psi = (self.cfg.gamma_d / self.cfg.v_th) * torch.clamp(1.0 - torch.abs((v_next_raw - b_next) / self.cfg.v_th), min=0.0)

        is_currently_refractory = self.ref_counts > 0
        spike_cond = (v_next_raw > b_next) & (~is_currently_refractory)
        self.z = spike_cond.float()

        v_next = torch.where(is_currently_refractory, self.v, v_next_raw - self.z * b_next)

        self.ref_counts = torch.clamp(self.ref_counts - self.cfg.time_step_dt, min=0.0)
        self.ref_counts = torch.where(self.z > 0, torch.tensor(self.cfg.t_ref, device=self.device, dtype=torch.float32), self.ref_counts)

        # 二重指数フィルタ
        self.z_bar_bar = self.decay_d * self.z_bar_bar + self.q
        self.q = self.decay_r * self.q + self.c_rd * self.z
        
        # 低域通過活動
        self.z_bar = self.alpha * self.z_bar + self.z_bar_bar

        self.v = v_next
        self.b = b_next


class LIFGroup(BaseNeuronGroup):
    def __init__(self, n_neurons, cfg: Config, **kwargs):
        super().__init__(n_neurons, cfg, **kwargs)
        self.beta = 0.0

    def step(self, input_current):
        v_next_raw = self.alpha * self.v + input_current
        b_next = torch.full((self.n,), self.cfg.v_th, device=self.device, dtype=torch.float32)
        self._update_spikes_and_filters(v_next_raw, b_next)


class ALIFGroup(BaseNeuronGroup):
    def __init__(self, n_neurons, cfg: Config, **kwargs):
        super().__init__(n_neurons, cfg, **kwargs)
        self.beta = cfg.beta
        self.rho = float(np.exp(-cfg.time_step_dt / cfg.tau_alif))
        self.a = torch.zeros(self.n, device=self.device, dtype=torch.float32)

    def reset_state(self):
        super().reset_state()
        self.a = torch.zeros(self.n, device=self.device, dtype=torch.float32)

    def step(self, input_current):
        self.a = self.rho * self.a + self.z
        v_next_raw = self.alpha * self.v + input_current
        b_next = self.cfg.v_th + self.beta * self.a
        self._update_spikes_and_filters(v_next_raw, b_next)


# ==========================================
# 最適化手法 (Predictive E-prop)
# ==========================================
class EPropOptimizer:
    def __init__(self, cfg: Config, decay_r: float, decay_d: float, c_rd: float, device=device):
        self.cfg = cfg
        self.device = device

        self.decay_r = decay_r
        self.decay_d = decay_d
        self.c_rd = c_rd

        n_neurons = cfg.n_neurons
        n_outputs = cfg.n_outputs

        self.e_trace = torch.zeros((n_neurons, n_neurons), device=self.device, dtype=torch.float32)
        self.eps_b = torch.zeros((n_neurons, n_neurons), device=self.device, dtype=torch.float32)

        self.q = torch.zeros((n_neurons, n_neurons), device=self.device, dtype=torch.float32)
        self.e_bar_bar = torch.zeros((n_neurons, n_neurons), device=self.device, dtype=torch.float32)

        self.grad_w_rec = torch.zeros((n_neurons, n_neurons), device=self.device, dtype=torch.float32)
        self.grad_w_out = torch.zeros((n_outputs, n_neurons), device=self.device, dtype=torch.float32)

        self.grad_w_rec_accumulated = torch.zeros((n_neurons, n_neurons), device=self.device, dtype=torch.float32)
        self.grad_w_out_accumulated = torch.zeros((n_outputs, n_neurons), device=self.device, dtype=torch.float32)

        self.spike_buffer = torch.zeros(n_neurons, device=self.device, dtype=torch.float32)
        self.e_trace_sum_t_delays = torch.zeros((n_neurons, n_neurons), device=self.device, dtype=torch.float32)
        self.step_counter = 0

    def reset_traces(self):
        self.e_trace.zero_()
        self.eps_b.zero_()

        self.q.zero_()
        self.e_bar_bar.zero_()

        self.grad_w_rec.zero_()
        self.grad_w_out.zero_()
        self.grad_w_rec_accumulated.zero_()
        self.grad_w_out_accumulated.zero_()

        self.spike_buffer.zero_()
        self.e_trace_sum_t_delays.zero_()
        self.step_counter = 0

    def update_traces_and_gradients(self, neurons, z_bar_curr, z_bar_prev, z_bar_prev_prev, psi_prev, L_t, error_t):
        for idx, neuron_group in enumerate(neurons):
            offset = 0 if idx == 0 else neurons[0].n
            n_g = neuron_group.n
            beta = neuron_group.beta
            rho = getattr(neuron_group, "rho", 0.0)

            psi_curr = neuron_group.psi.unsqueeze(1)
            psi_prev_g = psi_prev[offset:offset + n_g].unsqueeze(1)

            if beta > 0:
                self.eps_b[offset:offset + n_g, :] = psi_prev_g * z_bar_prev_prev.unsqueeze(0) + (rho - beta * psi_prev_g) * self.eps_b[offset:offset + n_g, :]

            self.e_trace[offset:offset + n_g, :] = psi_curr * (z_bar_prev.unsqueeze(0) - beta * self.eps_b[offset:offset + n_g, :])

        self.e_bar_bar = self.decay_d * self.e_bar_bar + self.q
        self.q = self.decay_r * self.q + self.e_trace

        self.grad_w_rec_accumulated += L_t.unsqueeze(1) * self.e_bar_bar

        self.grad_w_out_accumulated += torch.outer(error_t, z_bar_curr)

        all_spikes = torch.cat([g.z for g in neurons])
        self.spike_buffer += all_spikes
        self.e_trace_sum_t_delays += self.e_trace

        self.step_counter += 1

        if self.step_counter >= self.cfg.t_delay:
            self.grad_w_rec = self.grad_w_rec_accumulated.clone()
            self.grad_w_out = self.grad_w_out_accumulated.clone()

            f_bar = self.spike_buffer / self.cfg.t_delay
            reg_signal = (self.cfg.lambda_reg / self.cfg.t_delay) * (f_bar - self.cfg.f_star)

            self.grad_w_rec += reg_signal.unsqueeze(1) * self.e_trace_sum_t_delays

            self.spike_buffer.zero_()
            self.e_trace_sum_t_delays.zero_()
            self.grad_w_rec_accumulated.zero_()
            self.grad_w_out_accumulated.zero_()
            self.step_counter = 0

            return True

        return False

    def apply_weight_update(self, w_rec, w_out):
        w_rec -= self.cfg.eta_rec * self.grad_w_rec + self.cfg.eta_rec * (2.0 * self.cfg.lambda_w) * w_rec
        w_out -= self.cfg.eta_out * self.grad_w_out + self.cfg.eta_out * (2.0 * self.cfg.lambda_w) * w_out

        self.grad_w_rec.zero_()
        self.grad_w_out.zero_()

        return w_rec, w_out


# ==========================================
# ネットワーク全体モデル
# ==========================================
class PredictiveEPropNet:
    def __init__(self, cfg: Config, device=device):
        self.cfg = cfg
        self.device = device

        self.lif = LIFGroup(cfg.n_lif, cfg=cfg, device=device)
        self.alif = ALIFGroup(cfg.n_alif, cfg=cfg, device=device)
        self.neurons = [self.lif, self.alif]

        self.w_in = torch.randn(cfg.n_neurons, cfg.n_inputs, device=device, dtype=torch.float32) / np.sqrt(cfg.g)
        self.w_fb = torch.randn(cfg.n_neurons, cfg.n_outputs, device=device, dtype=torch.float32) / np.sqrt(cfg.g)

        self.w_rec_mask = (torch.rand(cfg.n_neurons, cfg.n_neurons, device=device) < cfg.density).float()
        self.w_rec_mask.fill_diagonal_(0.0)
        self.w_rec = (torch.randn(cfg.n_neurons, cfg.n_neurons, device=device, dtype=torch.float32) / np.sqrt(cfg.n_neurons)) * self.w_rec_mask

        self.w_out = torch.randn(cfg.n_outputs, cfg.n_neurons, device=device, dtype=torch.float32) / np.sqrt(cfg.n_neurons)
        self.B = torch.randn(cfg.n_neurons, cfg.n_outputs, device=device, dtype=torch.float32) / np.sqrt(cfg.n_neurons)

        self.s = torch.zeros(cfg.n_neurons, device=device, dtype=torch.float32)

        self.optimizer = EPropOptimizer(cfg=self.cfg, decay_r=self.lif.decay_r, decay_d=self.lif.decay_d, c_rd=self.lif.c_rd, device=device)

        self.rec_z = None
        self.rec_L = None
        self.rec_e = None
        self.rec_f = None

    def reset_state(self):
        self.lif.reset_state()
        self.alif.reset_state()
        self.optimizer.reset_traces()
        self.s.zero_()

    def generate_ou_noise(self):
        decay = float(np.exp(-self.cfg.time_step_dt / self.cfg.tau_s))
        self.s = self.cfg.mu_s + (self.s - self.cfg.mu_s) * decay + self.cfg.sigma_s * (self.cfg.time_step_dt / self.cfg.tau_s) * torch.randn(self.cfg.n_neurons, device=self.device, dtype=torch.float32)
        return self.s

    @torch.no_grad()
    def run_full_epoch(self, target_signal, record=False):
        self.reset_state()

        timesteps = len(target_signal)
        outputs = torch.zeros((timesteps, self.cfg.n_outputs), device=self.device, dtype=torch.float32)

        z_bar_prev = torch.zeros(self.cfg.n_neurons, device=self.device, dtype=torch.float32)
        z_bar_prev_prev = torch.zeros(self.cfg.n_neurons, device=self.device, dtype=torch.float32)
        psi_prev = torch.zeros(self.cfg.n_neurons, device=self.device, dtype=torch.float32)

        if record:
            self.rec_z = torch.zeros((timesteps, self.cfg.n_neurons), device=self.device, dtype=torch.bool)
            self.rec_L = torch.zeros(timesteps, device=self.device, dtype=torch.float32)
            self.rec_e = torch.zeros(timesteps, device=self.device, dtype=torch.float32)
            self.rec_f = torch.zeros(timesteps, device=self.device, dtype=torch.float32)

        train_cutoff = self.cfg.training_phase
        error_cutoff = self.cfg.training_phase + self.cfg.error_driven_phase

        for t in range(timesteps):
            if t < train_cutoff:
                phase = "training"
                xi = 1.0
            elif t < error_cutoff:
                phase = "error-driven"
                xi = 1.0
            else:
                phase = "free-running"
                xi = 0.0

            x_t = target_signal[t]

            all_z_bar_bar = torch.cat([self.lif.z_bar_bar, self.alif.z_bar_bar])
            all_z_bar = torch.cat([self.lif.z_bar, self.alif.z_bar])

            y_t = self.w_out @ all_z_bar_bar
            outputs[t] = y_t

            d_t = y_t - x_t

            i_rec = self.w_rec @ all_z_bar_bar
            i_in = self.w_in @ d_t
            i_fb = self.w_fb @ y_t

            s_t = self.generate_ou_noise()

            total_current = xi * i_in + i_rec + i_fb + self.cfg.I_bias # + s_t

            self.lif.step(total_current[:self.cfg.n_lif])
            self.alif.step(total_current[self.cfg.n_lif:])

            if phase == "training":
                L_t = self.B @ d_t
                all_psi = torch.cat([self.lif.psi, self.alif.psi])

                should_update = self.optimizer.update_traces_and_gradients(self.neurons, all_z_bar, z_bar_prev, z_bar_prev_prev, psi_prev, L_t, d_t)

                if record:
                    current_spikes = torch.cat([self.lif.z, self.alif.z])
                    self.rec_z[t] = current_spikes > 0
                    self.rec_L[t] = L_t[0]
                    self.rec_e[t] = torch.mean(torch.abs(self.optimizer.e_trace))
                    self.rec_f[t] = torch.mean(current_spikes) * 1000.0

                if should_update:
                    self.w_rec, self.w_out = self.optimizer.apply_weight_update(self.w_rec, self.w_out)
                    self.w_rec *= self.w_rec_mask

                z_bar_prev_prev = z_bar_prev.clone()
                z_bar_prev = all_z_bar.clone()
                psi_prev = all_psi.clone()

            else:
                if record:
                    current_spikes = torch.cat([self.lif.z, self.alif.z])
                    self.rec_z[t] = current_spikes > 0

                    self.optimizer.spike_buffer += current_spikes
                    self.optimizer.step_counter += 1

                    if self.optimizer.step_counter >= self.optimizer.cfg.t_delay:
                        self.optimizer.spike_buffer.zero_()
                        self.optimizer.step_counter = 0

                    self.rec_f[t] = torch.mean(current_spikes) * 1000.0

        return outputs


# ==========================================
# タスク実行とグラフ描画
# ==========================================
if __name__ == "__main__":
    # 設定クラスのインスタンス化 (ここで全てのハイパーパラメータを調整可能)
    cfg = Config()

    t_all = np.arange(0, cfg.total_time, cfg.time_step_dt)

    def sine_wave(t_array):
        return cfg.A * np.sin((2.0 * np.pi / cfg.T) * t_array + cfg.phi) + cfg.c

    target_all_np = sine_wave(t_all).reshape(-1, cfg.n_inputs).astype(np.float32)
    target_all = torch.tensor(target_all_np, device=device)

    net = PredictiveEPropNet(cfg=cfg, device=device)

    out_all_torch = None
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output", exist_ok=True)

    len_train = cfg.training_phase
    len_error = cfg.error_driven_phase

    for epoch in range(cfg.epochs):
        print(f"--- Epoch {epoch + 1}/{cfg.epochs} ---")

        do_record = (epoch + 1) % 10 == 0
        out_all_torch = net.run_full_epoch(target_all, record=do_record)

        loss_train = torch.mean((out_all_torch[:len_train] - target_all[:len_train]) ** 2).item()
        loss_error = torch.mean((out_all_torch[len_train:len_train + len_error] - target_all[len_train:len_train + len_error]) ** 2).item()
        loss_free = torch.mean((out_all_torch[len_train + len_error:] - target_all[len_train + len_error:]) ** 2).item()

        print(f"  Training Loss: {loss_train:.5f}")
        print(f"  Error-driven Loss: {loss_error:.5f}")
        print(f"  Free-running Loss: {loss_free:.5f}")

        if do_record:
            fig_analysis, axes_analysis = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

            spikes = net.rec_z.cpu().numpy()
            y_idx, x_idx = np.where(spikes.T)

            axes_analysis[0].scatter(x_idx, y_idx, s=1, color="black", alpha=0.5)
            axes_analysis[0].set_title(f"Spikes z (Epoch {epoch + 1})")
            axes_analysis[0].set_ylabel("Neuron index")

            axes_analysis[1].plot(t_all, net.rec_L.cpu().numpy(), color="blue")
            axes_analysis[1].set_title("Learning Signal L (Neuron 0)")
            axes_analysis[1].set_ylabel("L")

            axes_analysis[2].plot(t_all, net.rec_e.cpu().numpy(), color="red")
            axes_analysis[2].set_title("Eligibility Trace e (Mean Absolute Value)")
            axes_analysis[2].set_ylabel("e")

            window = 100
            smoothed_f = np.convolve(net.rec_f.cpu().numpy(), np.ones(window) / window, mode="same")

            axes_analysis[3].plot(t_all, smoothed_f, color="green")
            axes_analysis[3].set_title("Mean Firing Rate (100ms Moving Average)")
            axes_analysis[3].set_xlabel("Time [ms]")
            axes_analysis[3].set_ylabel("Hz")

            plt.tight_layout()

            analysis_filename = f"output/analysis_epoch_{epoch + 1}_{timestamp}.png"
            plt.savefig(analysis_filename)
            plt.close(fig_analysis)

            print(f"  => 内部状態のプロットを '{analysis_filename}' に保存しました。")

    out_all = out_all_torch.cpu().numpy()
    target_all = target_all_np

    img_filename = f"output/predictive_eprop_result_{timestamp}.png"

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    axes[0].plot(t_all[:len_train], target_all[:len_train], label="Target", color="black", linestyle="--")
    axes[0].plot(t_all[:len_train], out_all[:len_train], label="Output", color="blue", alpha=0.7)
    axes[0].set_title("Training Phase")
    axes[0].set_ylabel("Signal")
    axes[0].legend(loc="upper right")

    axes[1].plot(t_all[len_train:len_train + len_error], target_all[len_train:len_train + len_error], label="Target", color="black", linestyle="--")
    axes[1].plot(t_all[len_train:len_train + len_error], out_all[len_train:len_train + len_error], label="Output", color="orange", alpha=0.7)
    axes[1].set_title("Error-driven Phase")
    axes[1].set_ylabel("Signal")
    axes[1].legend(loc="upper right")

    axes[2].plot(t_all[len_train + len_error:], target_all[len_train + len_error:], label="Target", color="black", linestyle="--")
    axes[2].plot(t_all[len_train + len_error:], out_all[len_train + len_error:], label="Output", color="green", alpha=0.7)
    axes[2].set_title("Free-running Phase")
    axes[2].set_xlabel("Time [ms]")
    axes[2].set_ylabel("Signal")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(img_filename)

    print(f"\n推論結果のプロットを画像ファイル '{img_filename}' として保存しました。")

    # ハイパーパラメータ一覧のテキスト出力 (Configから自動生成)
    param_filename = f"output/predictive_eprop_params_{timestamp}.txt"
    actual_params = asdict(cfg)

    with open(param_filename, "w", encoding="utf-8") as f:
        f.write("スクリプト内で使用されたハイパーパラメータ一覧\n")
        f.write("=" * 60 + "\n")
        for key, value in actual_params.items():
            f.write(f"{key}: {value}\n")

    print(f"スクリプトで使用されたパラメータ数値をテキストファイル '{param_filename}' として保存しました。")