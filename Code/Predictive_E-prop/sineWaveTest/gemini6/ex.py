import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import datetime
import os

# ==========================================
# ハイパーパラメータの集約 (dataclass)
# ==========================================
@dataclass
class Config:
    # シミュレーションフェーズ[cite: 76]
    dt: float = 1.0
    len_train: int = 10000
    len_error: int = 5000
    len_free: int = 5000
    epochs: int = 50
    
    # ターゲット信号 (サイン波)[cite: 76]
    A: float = 0.4
    T: float = 1000.0
    phi: float = 0.0
    c: float = 0.0
    
    # ネットワーク構造[cite: 76]
    n_inputs: int = 1
    n_outputs: int = 1
    n_lif: int = 100
    n_alif: int = 200
    g: float = 150.0
    density: float = 0.01  # sparsity = 0.99
    
    # ニューロンパラメーター[cite: 76]
    v_th: float = 0.6
    tau_mem: float = 20.0
    tau_r: float = 50.0
    tau_d: float = 125.0
    t_ref: float = 5.0
    gamma_d: float = 0.3
    tau_alif: float = 2000.0
    beta_alif: float = 0.01
    I_bias: float = 0.02
    
    # ノイズ (OUプロセス)[cite: 76]
    tau_s: float = 250.0
    sigma_s: float = 1.0
    mu_s: float = 0.0
    
    # E-prop オプティマイザ[cite: 76]
    eta_rec: float = 0.0004
    eta_out: float = 0.0004
    lambda_reg: float = 2.0
    lambda_w: float = 5e-6
    t_delay: int = 10
    f_star: float = 0.01

    # デバイス設定
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# ニューロンモデル (PyTorch / GPU対応)
# ==========================================
class BaseNeuronGroup:
    def __init__(self, cfg: Config, n_neurons: int, beta: float):
        self.cfg = cfg
        self.n = n_neurons
        self.beta = beta
        self.device = torch.device(cfg.device)

        self.alpha = torch.exp(torch.tensor(-cfg.dt / cfg.tau_mem, device=self.device))
        self.decay_r = torch.exp(torch.tensor(-cfg.dt / cfg.tau_r, device=self.device))
        self.decay_d = torch.exp(torch.tensor(-cfg.dt / cfg.tau_d, device=self.device))
        self.rho = torch.exp(torch.tensor(-cfg.dt / cfg.tau_alif, device=self.device))

        self.c_rd = 1.0 / (cfg.tau_r * cfg.tau_d)

        self.reset_state()

    def reset_state(self):
        self.v = torch.zeros(self.n, device=self.device)
        self.z = torch.zeros(self.n, device=self.device)
        self.z_bar = torch.zeros(self.n, device=self.device)
        self.z_bar_bar = torch.zeros(self.n, device=self.device)

        # 順方向の二重フィルタ状態
        self.q = torch.full((self.n,), self.c_rd, device=self.device)

        self.psi = torch.zeros(self.n, device=self.device)
        self.b = torch.full((self.n,), self.cfg.v_th, device=self.device)
        self.ref_counts = torch.zeros(self.n, device=self.device)
        
        self.a = torch.zeros(self.n, device=self.device)

    def _update_spikes_and_filters(self, v_next_raw, b_next):
        # 疑似微分 ψ の計算[cite: 54, 55]
        self.psi = (self.cfg.gamma_d / self.cfg.v_th) * torch.relu(1.0 - torch.abs((v_next_raw - b_next) / self.cfg.v_th))

        is_currently_refractory = self.ref_counts > 0

        # スパイク判定
        self.z = torch.where((v_next_raw > b_next) & (~is_currently_refractory), 
                             torch.tensor(1.0, device=self.device), 
                             torch.tensor(0.0, device=self.device))

        # 膜電位の更新とリセット
        v_next = v_next_raw - self.z * b_next
        v_next[is_currently_refractory] = self.v[is_currently_refractory]

        # 不応期の更新
        self.ref_counts = torch.clamp(self.ref_counts - self.cfg.dt, min=0.0)
        self.ref_counts[self.z > 0] = self.cfg.t_ref

        # 二重指数フィルタの更新[cite: 53]
        q_prev = self.q.clone()
        self.q = self.decay_r * self.q + self.c_rd * self.z
        self.z_bar_bar = self.decay_d * self.z_bar_bar + q_prev

        # 論文の定義に合わせた低域通過活動[cite: 55]
        self.z_bar = self.alpha * self.z_bar + self.z_bar_bar

        self.v = v_next
        self.b = b_next

class LIFGroup(BaseNeuronGroup):
    def __init__(self, cfg: Config):
        super().__init__(cfg, cfg.n_lif, 0.0)

    def step(self, input_current):
        v_next_raw = self.alpha * self.v + input_current
        b_next = torch.full((self.n,), self.cfg.v_th, device=self.device)
        self._update_spikes_and_filters(v_next_raw, b_next)

class ALIFGroup(BaseNeuronGroup):
    def __init__(self, cfg: Config):
        super().__init__(cfg, cfg.n_alif, cfg.beta_alif)

    def step(self, input_current):
        self.a = self.rho * self.a + self.z
        v_next_raw = self.alpha * self.v + input_current
        b_next = self.cfg.v_th + self.beta * self.a
        self._update_spikes_and_filters(v_next_raw, b_next)

# ==========================================
# 最適化手法 (Predictive E-prop)
# ==========================================
class EPropOptimizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.n_neurons = cfg.n_lif + cfg.n_alif
        self.n_outputs = cfg.n_outputs
        
        self.decay_r = torch.exp(torch.tensor(-cfg.dt / cfg.tau_r, device=self.device))
        self.decay_d = torch.exp(torch.tensor(-cfg.dt / cfg.tau_d, device=self.device))
        
        self.reset_traces()

    def reset_traces(self):
        self.e_trace = torch.zeros((self.n_neurons, self.n_neurons), device=self.device)
        self.eps_b = torch.zeros((self.n_neurons, self.n_neurons), device=self.device)

        self.eps_d = torch.zeros((self.n_neurons, self.n_neurons), device=self.device)
        self.eps_r = torch.zeros((self.n_neurons, self.n_neurons), device=self.device)
        self.e_filter = torch.zeros((self.n_neurons, self.n_neurons), device=self.device)

        self.grad_w_rec = torch.zeros((self.n_neurons, self.n_neurons), device=self.device)
        self.grad_w_out = torch.zeros((self.n_outputs, self.n_neurons), device=self.device)

        self.grad_w_rec_accumulated = torch.zeros((self.n_neurons, self.n_neurons), device=self.device)
        self.grad_w_out_accumulated = torch.zeros((self.n_outputs, self.n_neurons), device=self.device)

        self.spike_buffer = torch.zeros(self.n_neurons, device=self.device)
        self.e_trace_sum_t_delays = torch.zeros((self.n_neurons, self.n_neurons), device=self.device)

        self.step_counter = 0

    def update_traces_and_gradients(self, neurons, z_bar_curr, z_bar_prev, z_bar_prev_prev, psi_prev, L_t, error_t):
        # ------------------------------------------
        # 局所適格度 e_trace の更新[cite: 54, 55]
        # ------------------------------------------
        for idx, neuron_group in enumerate(neurons):
            offset = 0 if idx == 0 else neurons[0].n
            n_g = neuron_group.n
            beta = neuron_group.beta
            rho = getattr(neuron_group, "rho", 0.0)

            psi_curr = neuron_group.psi.unsqueeze(1)
            psi_prev_g = psi_prev[offset:offset + n_g].unsqueeze(1)

            if beta > 0:
                self.eps_b[offset:offset + n_g, :] = psi_prev_g * z_bar_prev_prev + (rho - beta * psi_prev_g) * self.eps_b[offset:offset + n_g, :]

            self.e_trace[offset:offset + n_g, :] = psi_curr * (z_bar_prev - beta * self.eps_b[offset:offset + n_g, :])

        # ------------------------------------------
        # 再帰重みの予測誤差勾配[cite: 53]
        # ------------------------------------------
        self.grad_w_rec_accumulated += L_t.unsqueeze(1) * self.e_filter

        self.eps_d = self.decay_d * self.eps_d + self.e_trace
        self.eps_r = self.decay_r * self.eps_r + self.e_trace

        self.e_filter = self.eps_d - self.eps_r

        # 出力重みの勾配[cite: 55]
        self.grad_w_out_accumulated += torch.ger(error_t, z_bar_curr)

        # ------------------------------------------
        # ホメオスタシス正則化用の集積
        # ------------------------------------------
        all_spikes = torch.cat([g.z for g in neurons])
        self.spike_buffer += all_spikes
        self.e_trace_sum_t_delays += self.e_trace

        self.step_counter += 1

        # ------------------------------------------
        # t_delay ごとに重み更新用の勾配を確定
        # ------------------------------------------
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
        w_rec -= (self.cfg.eta_rec * self.grad_w_rec + self.cfg.eta_rec * (2.0 * self.cfg.lambda_w) * w_rec)
        w_out -= (self.cfg.eta_out * self.grad_w_out + self.cfg.eta_out * (2.0 * self.cfg.lambda_w) * w_out)
        
        self.grad_w_rec.zero_()
        self.grad_w_out.zero_()

        return w_rec, w_out

# ==========================================
# ネットワーク全体モデル
# ==========================================
class PredictiveEPropNet:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.n_neurons = cfg.n_lif + cfg.n_alif
        
        self.lif = LIFGroup(cfg)
        self.alif = ALIFGroup(cfg)
        self.neurons = [self.lif, self.alif]

        # 初期重みの設定[cite: 76]
        self.w_in = torch.randn(self.n_neurons, cfg.n_inputs, device=self.device) / (cfg.g ** 0.5)
        self.w_fb = torch.randn(self.n_neurons, cfg.n_outputs, device=self.device) / (cfg.g ** 0.5)

        self.w_rec_mask = (torch.rand(self.n_neurons, self.n_neurons, device=self.device) < cfg.density).float()
        self.w_rec_mask.fill_diagonal_(0.0)

        self.w_rec = (torch.randn(self.n_neurons, self.n_neurons, device=self.device) / (self.n_neurons ** 0.5)) * self.w_rec_mask
        self.w_out = torch.randn(cfg.n_outputs, self.n_neurons, device=self.device) / (self.n_neurons ** 0.5)
        
        # Broadcast weights[cite: 53]
        self.B = torch.randn(self.n_neurons, cfg.n_outputs, device=self.device) / (self.n_neurons ** 0.5)

        self.s = torch.zeros(self.n_neurons, device=self.device)
        self.optimizer = EPropOptimizer(cfg)

    def reset_state(self):
        self.lif.reset_state()
        self.alif.reset_state()
        self.optimizer.reset_traces()
        self.s.zero_()

    def generate_ou_noise(self):
        theta = self.cfg.dt / self.cfg.tau_s
        self.s = self.s + (self.cfg.mu_s - self.s) * theta + self.cfg.sigma_s * theta * torch.randn(self.n_neurons, device=self.device)
        return self.s

    def run_full_epoch(self, target_signal):
        self.reset_state()

        timesteps = target_signal.shape[0]
        outputs = torch.zeros((timesteps, self.cfg.n_outputs), device=self.device)

        z_bar_prev = torch.zeros(self.n_neurons, device=self.device)
        z_bar_prev_prev = torch.zeros(self.n_neurons, device=self.device)
        psi_prev = torch.zeros(self.n_neurons, device=self.device)

        spikes_train = 0.0
        spikes_error = 0.0
        spikes_free = 0.0

        # Autogradを無効化して高速化
        with torch.no_grad():
            for t in range(timesteps):
                # 3つのフェーズ制御[cite: 55, 56, 76]
                if t < self.cfg.len_train:
                    phase = "training"
                    xi = 1.0
                elif t < self.cfg.len_train + self.cfg.len_error:
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

                s_t = self.generate_ou_noise()

                i_rec = self.w_rec @ all_z_bar_bar
                i_in = self.w_in @ d_t
                i_fb = self.w_fb @ y_t

                total_current = xi * i_in + i_rec + i_fb + self.cfg.I_bias + s_t
                
                # ニューロン状態の更新
                self.lif.step(total_current[:self.cfg.n_lif])
                self.alif.step(total_current[self.cfg.n_lif:])

                current_spikes = torch.cat([self.lif.z, self.alif.z])
                spike_count = current_spikes.sum().item()

                if phase == "training":
                    spikes_train += spike_count
                    L_t = self.B @ d_t
                    all_psi = torch.cat([self.lif.psi, self.alif.psi])

                    should_update = self.optimizer.update_traces_and_gradients(
                        self.neurons, all_z_bar, z_bar_prev, z_bar_prev_prev, psi_prev, L_t, d_t
                    )

                    if should_update:
                        self.w_rec, self.w_out = self.optimizer.apply_weight_update(self.w_rec, self.w_out)
                        self.w_rec *= self.w_rec_mask

                    z_bar_prev_prev = z_bar_prev.clone()
                    z_bar_prev = all_z_bar.clone()
                    psi_prev = all_psi.clone()

                elif phase == "error-driven":
                    spikes_error += spike_count
                else:
                    spikes_free += spike_count

        # 平均発火率 (Hz) の計算: 合計スパイク数 / (ニューロン数 * 該当フェーズの秒数)
        hz_train = (spikes_train / (self.cfg.len_train * self.n_neurons)) * (1000.0 / self.cfg.dt)
        hz_error = (spikes_error / (self.cfg.len_error * self.n_neurons)) * (1000.0 / self.cfg.dt)
        hz_free = (spikes_free / (self.cfg.len_free * self.n_neurons)) * (1000.0 / self.cfg.dt)

        return outputs.cpu().numpy(), hz_train, hz_error, hz_free

# ==========================================
# タスク実行とグラフ描画
# ==========================================
if __name__ == "__main__":
    cfg = Config()
    
    total_time = cfg.len_train + cfg.len_error + cfg.len_free
    t_all = np.arange(0, total_time, 1)

    def sine_wave(t_array, cfg):
        return cfg.A * np.sin((2.0 * np.pi / cfg.T) * t_array + cfg.phi) + cfg.c

    target_numpy = sine_wave(t_all, cfg).reshape(-1, 1)
    target_all = torch.tensor(target_numpy, dtype=torch.float32, device=cfg.device)

    net = PredictiveEPropNet(cfg)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output", exist_ok=True)

    print(f"Device: {cfg.device}")
    
    fr_train_list, fr_error_list, fr_free_list = [], [], []

    for epoch in range(cfg.epochs):
        print(f"--- Epoch {epoch + 1}/{cfg.epochs} ---")

        out_all, hz_train, hz_error, hz_free = net.run_full_epoch(target_all)

        fr_train_list.append(hz_train)
        fr_error_list.append(hz_error)
        fr_free_list.append(hz_free)

        loss_train = np.mean((out_all[:cfg.len_train] - target_numpy[:cfg.len_train]) ** 2)
        loss_error = np.mean((out_all[cfg.len_train:cfg.len_train + cfg.len_error] - target_numpy[cfg.len_train:cfg.len_train + cfg.len_error]) ** 2)
        loss_free = np.mean((out_all[cfg.len_train + cfg.len_error:] - target_numpy[cfg.len_train + cfg.len_error:]) ** 2)

        print(f"  Training Loss: {loss_train:.5f}")
        print(f"  Error-driven Loss: {loss_error:.5f}")
        print(f"  Free-running Loss: {loss_free:.5f}")

    # --- 発火率の推移のプロット (論文 図2C 再現)[cite: 60] ---
    plt.figure(figsize=(8, 5))
    epochs_arr = np.arange(1, cfg.epochs + 1)
    plt.plot(epochs_arr, fr_train_list, label="Training", color="blue")
    plt.plot(epochs_arr, fr_error_list, label="Error-driven", color="orange")
    plt.plot(epochs_arr, fr_free_list, label="Free-running", color="green")
    
    plt.xlabel("Epoch")
    plt.ylabel("Firing Rate [Hz]")
    plt.title("Average Firing Rate across Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    fr_filename = f"output/firing_rate_{timestamp}.png"
    plt.savefig(fr_filename)
    plt.close()
    print(f"=> ネットワーク全体の平均発火率の推移を '{fr_filename}' に保存しました。")

    # --- 最終エポックの推論結果のプロット ---
    img_filename = f"output/predictive_eprop_result_{timestamp}.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    axes[0].plot(t_all[:cfg.len_train], target_numpy[:cfg.len_train], label="Target", color="black", linestyle="--")
    axes[0].plot(t_all[:cfg.len_train], out_all[:cfg.len_train], label="Output", color="blue", alpha=0.7)
    axes[0].set_title("Training Phase")
    axes[0].set_ylabel("Signal")
    axes[0].legend(loc="upper right")

    axes[1].plot(t_all[cfg.len_train:cfg.len_train + cfg.len_error], target_numpy[cfg.len_train:cfg.len_train + cfg.len_error], label="Target", color="black", linestyle="--")
    axes[1].plot(t_all[cfg.len_train:cfg.len_train + cfg.len_error], out_all[cfg.len_train:cfg.len_train + cfg.len_error], label="Output", color="orange", alpha=0.7)
    axes[1].set_title("Error-driven Phase")
    axes[1].set_ylabel("Signal")
    axes[1].legend(loc="upper right")

    axes[2].plot(t_all[cfg.len_train + cfg.len_error:], target_numpy[cfg.len_train + cfg.len_error:], label="Target", color="black", linestyle="--")
    axes[2].plot(t_all[cfg.len_train + cfg.len_error:], out_all[cfg.len_train + cfg.len_error:], label="Output", color="green", alpha=0.7)
    axes[2].set_title("Free-running Phase")
    axes[2].set_xlabel("Time [ms]")
    axes[2].set_ylabel("Signal")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(img_filename)
    print(f"=> 推論結果のプロットを画像ファイル '{img_filename}' として保存しました。")

    # --- パラメータ保存 ---
    param_filename = f"output/predictive_eprop_params_{timestamp}.txt"
    with open(param_filename, "w", encoding="utf-8") as f:
        f.write("スクリプト内で使用されたハイパーパラメータ一覧\n")
        f.write("=" * 60 + "\n")
        for key, value in asdict(cfg).items():
            f.write(f"{key}: {value}\n")

    print(f"=> スクリプトで使用されたパラメータ数値をテキストファイル '{param_filename}' として保存しました。")