import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from dataclasses import dataclass
import math

# ==========================================
# ハイパーパラメータ設定
# ==========================================
@dataclass
class EPropConfig:
    # ニューロン・ネットワークのスケール
    n_inputs: int = 1
    n_lif: int = 100
    n_alif: int = 200
    n_outputs: int = 1
    
    # 時間・シミュレーション設定
    dt: float = 1.0
    len_train: int = 10000
    len_error: int = 5000
    len_free: int = 5000
    epochs: int = 50
    
    # ニューロンモデルパラメータ
    tau_mem: float = 20.0
    tau_r: float = 50.0
    tau_d: float = 125.0
    v_th: float = 0.6
    gamma_d: float = 0.3
    t_ref: float = 5.0
    tau_alif: float = 2000.0
    beta: float = 0.0174
    
    # ネットワーク結合パラメータ
    g: float = 150.0
    density: float = 0.01
    I_bias: float = 0.02
    
    # ノイズパラメータ (Ornstein-Uhlenbeck)
    tau_s: float = 250.0
    sigma_s: float = 1.0
    mu_s: float = 0.0
    
    # E-prop学習最適化パラメータ
    eta: float = 0.0004
    lambda_reg: float = 2.0
    lambda_w: float = 5e-6
    t_delay: int = 10
    f_star: float = 0.01
    
    # サイン波タスクターゲットパラメータ
    A: float = 0.4
    T: float = 1000.0
    phi: float = 0.0
    c: float = 0.0

    @property
    def n_neurons(self) -> int:
        return self.n_lif + self.n_alif
    
    @property
    def total_time(self) -> int:
        return self.len_train + self.len_error + self.len_free

# GPUデバイスの検出
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# ニューロンモデル
# ==========================================
class BaseNeuronGroup:
    def __init__(self, cfg: EPropConfig, n_neurons: int):
        self.cfg = cfg
        self.n = n_neurons
        
        # Eq. 1: alpha = exp(-dt / tau_mem)
        self.alpha = math.exp(-cfg.dt / cfg.tau_mem)
        # Eq. 2: decay constants for double exponential filter
        self.decay_r = math.exp(-cfg.dt / cfg.tau_r)
        self.decay_d = math.exp(-cfg.dt / cfg.tau_d)
        
        # Eq. 2: c_rd = 1 / (tau_r * tau_d)
        self.c_rd = 1.0 / (cfg.tau_r * cfg.tau_d)
        
        self.reset_state()

    def reset_state(self):
        self.v = torch.zeros(self.n, device=device)
        self.z = torch.zeros(self.n, device=device)
        self.z_bar = torch.zeros(self.n, device=device)
        self.z_bar_bar = torch.zeros(self.n, device=device)
        # Eq. 2: q^0 = 1 / (tau_r * tau_d)
        self.q = torch.full((self.n,), self.c_rd, device=device)
        self.psi = torch.zeros(self.n, device=device)
        self.b = torch.full((self.n,), self.cfg.v_th, device=device)
        self.ref_counts = torch.zeros(self.n, device=device)

    def _update_spikes_and_filters(self, v_next_raw, b_next):
        # Eq. 7: テンソル微分の近似 (Tent surrogate gradient)
        self.psi = (self.cfg.gamma_d / self.cfg.v_th) * torch.clamp(
            1.0 - torch.abs((v_next_raw - b_next) / self.cfg.v_th), min=0.0
        )
        
        # 不応期中の擬似微分を0にマスク
        is_currently_refractory = self.ref_counts > 0
        self.psi[is_currently_refractory] = 0.0
        
        # Eq. 1: z_i^t = theta(v_i^t - b_i^t)
        self.z = torch.where((v_next_raw > b_next) & (~is_currently_refractory), 1.0, 0.0)
        
        # Eq. 1: Repolarization mechanism
        v_next = v_next_raw - self.z * b_next
        
        # Section 2.1: 不応期の適用 (5 ms)
        v_next[is_currently_refractory] = self.v[is_currently_refractory]
        
        self.ref_counts = torch.clamp(self.ref_counts - self.cfg.dt, min=0.0)
        self.ref_counts[self.z > 0] = self.cfg.t_ref
        
        # Eq. 2: 2重指数関数フィルター
        self.z_bar_bar = self.decay_d * self.z_bar_bar + self.q
        self.q = self.decay_r * self.q + self.c_rd * self.z
        
        # Section 2.2: 単一低周波フィルター
        self.z_bar = self.alpha * self.z_bar + self.z_bar_bar

        self.v = v_next
        self.b = b_next

class LIFGroup(BaseNeuronGroup):
    def __init__(self, cfg: EPropConfig, n_neurons: int):
        super().__init__(cfg, n_neurons)
        self.beta = 0.0

    def step(self, input_current):
        v_next_raw = self.alpha * self.v + input_current
        b_next = torch.full((self.n,), self.cfg.v_th, device=device)
        self._update_spikes_and_filters(v_next_raw, b_next)

class ALIFGroup(BaseNeuronGroup):
    def __init__(self, cfg: EPropConfig, n_neurons: int):
        super().__init__(cfg, n_neurons)
        self.beta = cfg.beta
        self.rho = math.exp(-cfg.dt / cfg.tau_alif)
        self.a = torch.zeros(self.n, device=device)

    def reset_state(self):
        super().reset_state()
        self.a = torch.zeros(self.n, device=device)

    def step(self, input_current):
        self.a = self.rho * self.a + self.z
        v_next_raw = self.alpha * self.v + input_current
        b_next = self.cfg.v_th + self.beta * self.a
        self._update_spikes_and_filters(v_next_raw, b_next)

# ==========================================
# 最適化手法 (Predictive E-prop with Adam)
# ==========================================
class EPropOptimizer:
    def __init__(self, cfg: EPropConfig):
        self.cfg = cfg
        n = cfg.n_neurons
        nout = cfg.n_outputs
        
        self.e_trace = torch.zeros((n, n), device=device)
        self.eps_b = torch.zeros((n, n), device=device)
        
        self.grad_w_rec = torch.zeros((n, n), device=device)
        self.grad_w_out = torch.zeros((nout, n), device=device)
        self.grad_w_rec_accumulated = torch.zeros((n, n), device=device)
        self.grad_w_out_accumulated = torch.zeros((nout, n), device=device)
        
        self.spike_buffer = torch.zeros(n, device=device)
        self.e_trace_sum = torch.zeros((n, n), device=device)
        self.step_counter = 0

        # Adamパラメータ
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.adam_t = 0
        
        self.m_w_rec = torch.zeros((n, n), device=device)
        self.v_w_rec = torch.zeros((n, n), device=device)
        self.m_w_out = torch.zeros((nout, n), device=device)
        self.v_w_out = torch.zeros((nout, n), device=device)

    def reset_traces(self):
        self.e_trace.zero_()
        self.eps_b.zero_()
        self.grad_w_rec.zero_()
        self.grad_w_out.zero_()
        self.grad_w_rec_accumulated.zero_()
        self.grad_w_out_accumulated.zero_()
        self.spike_buffer.zero_()
        self.e_trace_sum.zero_()
        self.step_counter = 0

    def update_traces_and_gradients(self, neurons, z_bar_bar_curr, z_bar_prev, z_bar_prev_prev, psi_prev, L_t, error_t):
        for idx, neuron_group in enumerate(neurons):
            offset = 0 if idx == 0 else neurons[0].n
            n_g = neuron_group.n
            beta = neuron_group.beta
            rho = getattr(neuron_group, 'rho', 0.0)
            
            psi_curr = neuron_group.psi.unsqueeze(1)
            psi_prev_g = psi_prev[offset:offset+n_g].unsqueeze(1)
            
            if beta > 0:
                self.eps_b[offset:offset+n_g, :] = psi_prev_g * z_bar_prev_prev + \
                                                  (rho - beta * psi_prev_g) * self.eps_b[offset:offset+n_g, :]
            
            self.e_trace[offset:offset+n_g, :] = psi_curr * (z_bar_prev - beta * self.eps_b[offset:offset+n_g, :])
        
        # 勾配の蓄積
        self.grad_w_rec_accumulated -= L_t.unsqueeze(1) * self.e_trace
        # 修正箇所: error_tの次元が消滅しないよう .squeeze(-1) を .view(-1) に変更
        self.grad_w_out_accumulated += torch.outer(error_t.view(-1), z_bar_bar_curr)
        
        all_spikes = torch.cat([g.z for g in neurons])
        self.spike_buffer += all_spikes
        self.e_trace_sum += self.e_trace
        self.step_counter += 1
        
        if self.step_counter >= self.cfg.t_delay:
            self.grad_w_rec = self.grad_w_rec_accumulated.clone()
            self.grad_w_out = self.grad_w_out_accumulated.clone()

            # ホメオスタシス正則化
            f_bar = self.spike_buffer / self.cfg.t_delay
            reg_signal = (self.cfg.lambda_reg / self.cfg.t_delay) * (f_bar - self.cfg.f_star)
            self.grad_w_rec += reg_signal.unsqueeze(1) * self.e_trace_sum
            
            self.spike_buffer.zero_()
            self.e_trace_sum.zero_()
            self.step_counter = 0
            return True
        return False

    def apply_weight_update(self, w_rec, w_out):
        self.adam_t += 1
        
        g_w_rec = self.grad_w_rec + 2 * self.cfg.lambda_w * w_rec
        g_w_out = self.grad_w_out + 2 * self.cfg.lambda_w * w_out
        
        # Adam: w_rec
        self.m_w_rec = self.beta1 * self.m_w_rec + (1 - self.beta1) * g_w_rec
        self.v_w_rec = self.beta2 * self.v_w_rec + (1 - self.beta2) * (g_w_rec ** 2)
        m_hat_rec = self.m_w_rec / (1 - self.beta1 ** self.adam_t)
        v_hat_rec = self.v_w_rec / (1 - self.beta2 ** self.adam_t)
        w_rec -= self.cfg.eta * m_hat_rec / (torch.sqrt(v_hat_rec) + self.epsilon)

        # Adam: w_out
        self.m_w_out = self.beta1 * self.m_w_out + (1 - self.beta1) * g_w_out
        self.v_w_out = self.beta2 * self.v_w_out + (1 - self.beta2) * (g_w_out ** 2)
        m_hat_out = self.m_w_out / (1 - self.beta1 ** self.adam_t)
        v_hat_out = self.v_w_out / (1 - self.beta2 ** self.adam_t)
        w_out -= self.cfg.eta * m_hat_out / (torch.sqrt(v_hat_out) + self.epsilon)
        
        self.grad_w_rec.zero_()
        self.grad_w_out.zero_()
        return w_rec, w_out

# ==========================================
# ネットワーク全体モデル
# ==========================================
class PredictiveEPropNet:
    def __init__(self, cfg: EPropConfig):
        self.cfg = cfg
        
        self.lif = LIFGroup(cfg, cfg.n_lif)
        self.alif = ALIFGroup(cfg, cfg.n_alif)
        self.neurons = [self.lif, self.alif]
        
        n = cfg.n_neurons
        
        # W^in, W^fb ~ N(0, 1) / sqrt(g)
        self.w_in = torch.randn(n, cfg.n_inputs, device=device) / math.sqrt(cfg.g)
        self.w_fb = torch.randn(n, cfg.n_outputs, device=device) / math.sqrt(cfg.g)
        
        # W^rec (Sparsity = 0.99)
        self.w_rec_mask = (torch.rand(n, n, device=device) < cfg.density).float()
        self.w_rec_mask.fill_diagonal_(0.0)
        self.w_rec = (torch.randn(n, n, device=device) / math.sqrt(n)) * self.w_rec_mask
        
        # W^out ~ N(0, 1) / sqrt(N_rec)
        self.w_out = torch.randn(cfg.n_outputs, n, device=device) / math.sqrt(n)
        
        self.B = -self.w_in * math.sqrt(cfg.g)
        self.s = torch.full((n,), -10, device=device)
        
        self.optimizer = EPropOptimizer(cfg) 

    def reset_state(self):
        self.lif.reset_state()
        self.alif.reset_state()
        self.optimizer.reset_traces()
        self.s.zero_()

    def generate_ou_noise(self):
        decay = math.exp(-self.cfg.dt / self.cfg.tau_s)
        self.s = self.cfg.mu_s + (self.s - self.cfg.mu_s) * decay + \
                 self.cfg.sigma_s * (self.cfg.dt / self.cfg.tau_s) * torch.randn(self.cfg.n_neurons, device=device)
        return self.s

    @torch.no_grad()
    def run_full_epoch(self, target_signal: torch.Tensor):
        self.reset_state()
        timesteps = target_signal.shape[0]
        outputs = torch.zeros((timesteps, self.cfg.n_outputs), device=device)
        
        z_bar_prev = torch.zeros(self.cfg.n_neurons, device=device)
        z_bar_prev_prev = torch.zeros(self.cfg.n_neurons, device=device)
        psi_prev = torch.zeros(self.cfg.n_neurons, device=device)
        
        # 各フェーズにおけるスパイク数をカウントする辞書を初期化
        spike_counts = {"training": 0.0, "error-driven": 0.0, "free-running": 0.0}
        
        for t in range(timesteps):
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
            
            self.lif.step(total_current[:self.cfg.n_lif])
            self.alif.step(total_current[self.cfg.n_lif:])
            
            # 発火率計算用に全ニューロンのスパイクを合算
            current_spikes = self.lif.z.sum() + self.alif.z.sum()
            spike_counts[phase] += current_spikes.item()
            
            if phase == "training":
                L_t = self.B @ d_t * self.lif.c_rd
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
        
        # フェーズごとのネットワーク平均発火率 (Hz) を計算
        firing_rates = {
            "training": (spike_counts["training"] / (self.cfg.n_neurons * self.cfg.len_train)) * (1000.0 / self.cfg.dt),
            "error-driven": (spike_counts["error-driven"] / (self.cfg.n_neurons * self.cfg.len_error)) * (1000.0 / self.cfg.dt),
            "free-running": (spike_counts["free-running"] / (self.cfg.n_neurons * self.cfg.len_free)) * (1000.0 / self.cfg.dt),
        }
            
        return outputs, firing_rates

# ==========================================
# タスク実行とグラフ描画
# ==========================================
if __name__ == "__main__":
    # Configインスタンスの作成
    cfg = EPropConfig()
    
    t_all = torch.arange(0, cfg.total_time, cfg.dt, device=device)
    
    def sine_wave(t_array):
        return cfg.A * torch.sin((2 * math.pi / cfg.T) * t_array + cfg.phi) + cfg.c

    target_all = sine_wave(t_all).view(-1, 1)
    
    net = PredictiveEPropNet(cfg)
    out_all = None
    
    # 平均発火率の推移を記録するリスト
    rate_history_train = []
    rate_history_error = []
    rate_history_free = []
    
    for epoch in range(cfg.epochs):
        print(f"--- Epoch {epoch+1}/{cfg.epochs} ---")
        
        out_all, firing_rates = net.run_full_epoch(target_signal=target_all)
        
        loss_train = torch.mean((out_all[:cfg.len_train] - target_all[:cfg.len_train])**2).item()
        loss_error = torch.mean((out_all[cfg.len_train:cfg.len_train+cfg.len_error] - target_all[cfg.len_train:cfg.len_train+cfg.len_error])**2).item()
        loss_free = torch.mean((out_all[cfg.len_train+cfg.len_error:] - target_all[cfg.len_train+cfg.len_error:])**2).item()
        
        rate_history_train.append(firing_rates["training"])
        rate_history_error.append(firing_rates["error-driven"])
        rate_history_free.append(firing_rates["free-running"])
        
        print(f"  [Loss] Train: {loss_train:.5f} | Error-driven: {loss_error:.5f} | Free-running: {loss_free:.5f}")
        print(f"  [Rate] Train: {firing_rates['training']:.2f} Hz | Error-driven: {firing_rates['error-driven']:.2f} Hz | Free-running: {firing_rates['free-running']:.2f} Hz")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputnext", exist_ok=True)
    
    # テンソルをCPU上のNumPy配列に変換
    t_all_np = t_all.cpu().numpy()
    target_all_np = target_all.cpu().numpy()
    out_all_np = out_all.cpu().numpy()
    
    # --- 画像ファイルの出力 ---
    img_filename = f"outputNext/predictive_eprop_result_{timestamp}.png"
    # サブプロットを4つに変更 (サイン波生成結果 x 3フェーズ + 平均発火率 x 1)
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    
    t1, t2 = cfg.len_train, cfg.len_train + cfg.len_error
    
    axes[0].plot(t_all_np[:t1], target_all_np[:t1], label="Target", color="black", linestyle="--")
    axes[0].plot(t_all_np[:t1], out_all_np[:t1], label="Output", color="blue", alpha=0.7)
    axes[0].set_title("Training Phase")
    axes[0].set_ylabel("Signal")
    axes[0].legend(loc="upper right")
    
    axes[1].plot(t_all_np[t1:t2], target_all_np[t1:t2], label="Target", color="black", linestyle="--")
    axes[1].plot(t_all_np[t1:t2], out_all_np[t1:t2], label="Output", color="orange", alpha=0.7)
    axes[1].set_title("Error-driven Phase")
    axes[1].set_ylabel("Signal")
    axes[1].legend(loc="upper right")
    
    axes[2].plot(t_all_np[t2:], target_all_np[t2:], label="Target", color="black", linestyle="--")
    axes[2].plot(t_all_np[t2:], out_all_np[t2:], label="Output", color="green", alpha=0.7)
    axes[2].set_title("Free-running Phase")
    axes[2].set_xlabel("Time [ms]")
    axes[2].set_ylabel("Signal")
    axes[2].legend(loc="upper right")
    
    # 平均発火率の推移プロットの追加
    epochs_range = range(1, cfg.epochs + 1)
    axes[3].plot(epochs_range, rate_history_train, label="Training", color="blue")
    axes[3].plot(epochs_range, rate_history_error, label="Error-driven", color="orange")
    axes[3].plot(epochs_range, rate_history_free, label="Free-running", color="green")
    axes[3].set_title("Average Firing Rate across Epochs")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Firing Rate [Hz]")
    axes[3].legend(loc="lower right")
    axes[3].grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(img_filename)
    print(f"\nプロットを画像ファイル '{img_filename}' として保存しました。")
    
    # --- パラメータ出力 ---
    param_filename = f"outputNext/predictive_eprop_params_{timestamp}.txt"
    with open(param_filename, "w", encoding="utf-8") as f:
        f.write("スクリプト内で使用されたハイパーパラメータ一覧\n")
        f.write("=" * 60 + "\n")
        for key, value in cfg.__dict__.items():
            f.write(f"{key}: {value}\n")
            
    print(f"スクリプトで使用されたパラメータ数値をテキストファイル '{param_filename}' として保存しました。")