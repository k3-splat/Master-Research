import numpy as np
import matplotlib.pyplot as plt
import datetime

# ==========================================
# ニューロンモデル
# ==========================================
class BaseNeuronGroup:
    def __init__(self, n_neurons, dt=1.0, tau_mem=20.0, tau_r=50.0, tau_d=125.0, v_th=0.6, gamma_d=0.3, t_ref=5.0):
        self.n = n_neurons
        self.dt = dt
        self.alpha = np.exp(-dt / tau_mem)
        self.decay_r = np.exp(-dt / tau_r)
        self.decay_d = np.exp(-dt / tau_d)
        self.v_th = v_th
        self.gamma_d = gamma_d
        self.c_rd = 1.0 / (tau_r * tau_d)
        self.t_ref = t_ref
        
        self.reset_state()

    def reset_state(self):
        self.v = np.zeros(self.n)
        self.z = np.zeros(self.n)
        self.z_bar = np.zeros(self.n)
        self.z_bar_bar = np.zeros(self.n)
        # Eq 2 に準拠した初期化
        self.q = np.zeros(self.n) + self.c_rd
        self.psi = np.zeros(self.n)
        self.b = np.full(self.n, self.v_th)
        self.ref_counts = np.zeros(self.n)

    def _update_spikes_and_filters(self, v_next_raw, b_next):
        # テンソル微分の近似
        self.psi = (self.gamma_d / self.v_th) * np.maximum(0, 1.0 - np.abs((v_next_raw - b_next) / self.v_th))
        
        # スパイク判定 (不応期考慮)
        self.z = np.where((v_next_raw > b_next) & (self.ref_counts <= 0), 1.0, 0.0)
        
        # Repolarization
        v_next = v_next_raw - self.z * b_next
        
        # 不応期の適用
        is_refractory = self.ref_counts > 0
        v_next[is_refractory] = self.v[is_refractory]
        
        self.ref_counts = np.maximum(0, self.ref_counts - self.dt)
        self.ref_counts[self.z > 0] = self.t_ref
        
        # 2重指数関数フィルター (Eq 2通りに復元)
        q_prev = self.q.copy()
        self.q = self.decay_r * self.q + self.c_rd * self.z
        self.z_bar_bar = self.decay_d * self.z_bar_bar + q_prev
        
        # 単一低周波フィルター
        self.z_bar = self.alpha * self.z_bar + self.z_bar_bar

        self.v = v_next
        self.b = b_next

class LIFGroup(BaseNeuronGroup):
    def __init__(self, n_neurons, **kwargs):
        super().__init__(n_neurons, **kwargs)
        self.beta = 0.0

    def step(self, input_current):
        v_next_raw = self.alpha * self.v + input_current
        b_next = np.full(self.n, self.v_th)
        self._update_spikes_and_filters(v_next_raw, b_next)

class ALIFGroup(BaseNeuronGroup):
    def __init__(self, n_neurons, tau_alif=2000.0, beta=0.01, **kwargs):
        super().__init__(n_neurons, **kwargs)
        self.rho = np.exp(-self.dt / tau_alif)
        self.beta = beta
        self.a = np.zeros(self.n)

    def reset_state(self):
        super().reset_state()
        self.a = np.zeros(self.n)

    def step(self, input_current):
        self.a = self.rho * self.a + self.z
        v_next_raw = self.alpha * self.v + input_current
        b_next = self.v_th + self.beta * self.a
        self._update_spikes_and_filters(v_next_raw, b_next)

# ==========================================
# 最適化手法 (Predictive E-prop)
# ==========================================
class EPropOptimizer:
    def __init__(self, n_neurons, n_inputs, n_outputs, eta=0.0004, lambda_reg=2.0, lambda_w=5e-6, t_delay=10, f_star=0.01):
        self.n_neurons = n_neurons
        self.eta = eta
        self.lambda_reg = lambda_reg
        self.lambda_w = lambda_w
        self.t_delay = t_delay
        self.f_star = f_star
        
        self.e_trace = np.zeros((n_neurons, n_neurons))
        self.eps_b = np.zeros((n_neurons, n_neurons))
        
        self.grad_w_rec = np.zeros((n_neurons, n_neurons))
        self.grad_w_out = np.zeros((n_outputs, n_neurons))
        
        self.spike_buffer = np.zeros(n_neurons)
        self.e_trace_sum = np.zeros((n_neurons, n_neurons))
        self.step_counter = 0

    def reset_traces(self):
        self.e_trace.fill(0)
        self.eps_b.fill(0)
        self.grad_w_rec.fill(0)
        self.grad_w_out.fill(0)
        self.spike_buffer.fill(0)
        self.e_trace_sum.fill(0)
        self.step_counter = 0

    def update_traces_and_gradients(self, neurons, z_bar_prev, z_bar_prev_prev, psi_prev, L_t, error_t):
        for idx, neuron_group in enumerate(neurons):
            offset = 0 if idx == 0 else neurons[0].n
            n_g = neuron_group.n
            beta = neuron_group.beta
            rho = getattr(neuron_group, 'rho', 0.0)
            
            psi_curr = neuron_group.psi[:, None]
            psi_prev_g = psi_prev[offset:offset+n_g, None]
            
            if beta > 0:
                self.eps_b[offset:offset+n_g, :] = psi_prev_g * z_bar_prev_prev + \
                                                  (rho - beta * psi_prev_g) * self.eps_b[offset:offset+n_g, :]
                
            self.e_trace[offset:offset+n_g, :] = psi_curr * (z_bar_prev - beta * self.eps_b[offset:offset+n_g, :])
        
        self.grad_w_rec += L_t[:, None] * self.e_trace
        self.grad_w_out += np.outer(error_t, z_bar_prev)
        
        all_spikes = np.concatenate([g.z for g in neurons])
        self.spike_buffer += all_spikes
        self.e_trace_sum += self.e_trace
        self.step_counter += 1
        
        if self.step_counter >= self.t_delay:
            f_bar = self.spike_buffer / self.t_delay
            reg_signal = (self.lambda_reg / self.t_delay) * (f_bar - self.f_star)
            self.grad_w_rec += reg_signal[:, None] * self.e_trace_sum
            
            self.spike_buffer.fill(0)
            self.e_trace_sum.fill(0)
            self.step_counter = 0
            return True
        return False

    def apply_weight_update(self, w_rec, w_out):
        grad_rec_clipped = np.clip(self.grad_w_rec, -10.0, 10.0)
        grad_out_clipped = np.clip(self.grad_w_out, -10.0, 10.0)
        
        w_rec -= self.eta * grad_rec_clipped + self.eta * (2 * self.lambda_w) * w_rec
        w_out -= self.eta * grad_out_clipped + self.eta * (2 * self.lambda_w) * w_out
        
        self.grad_w_rec.fill(0)
        self.grad_w_out.fill(0)
        return w_rec, w_out

# ==========================================
# ネットワーク全体モデル
# ==========================================
class PredictiveEPropNet:
    def __init__(self, n_inputs, n_lif, n_alif, n_outputs, g=150.0):
        self.n_inputs = n_inputs
        self.n_lif = n_lif
        self.n_alif = n_alif
        self.n_neurons = n_lif + n_alif
        self.n_outputs = n_outputs
        
        self.lif = LIFGroup(n_lif)
        self.alif = ALIFGroup(n_alif)
        self.neurons = [self.lif, self.alif]
        
        self.w_in = np.random.randn(self.n_neurons, n_inputs) / np.sqrt(g)
        self.w_fb = np.random.randn(self.n_neurons, n_outputs) / np.sqrt(g)
        self.w_rec = np.random.randn(self.n_neurons, self.n_neurons) / np.sqrt(self.n_neurons)
        self.w_out = np.random.randn(n_outputs, self.n_neurons) / np.sqrt(self.n_neurons)
        self.B = np.random.randn(self.n_neurons, n_outputs)
        
        self.I_bias = 0.02
        self.tau_s = 250.0
        self.sigma_s = 1.0  # Table SI通りに復元
        self.s = np.zeros(self.n_neurons)
        
        self.optimizer = EPropOptimizer(self.n_neurons, n_inputs, n_outputs, eta=0.0004) # Table SI通りに復元

    def reset_state(self):
        self.lif.reset_state()
        self.alif.reset_state()
        self.optimizer.reset_traces()
        self.s = np.zeros(self.n_neurons)

    def generate_ou_noise(self, dt=1.0):
        self.s += - (self.s / self.tau_s) * dt + self.sigma_s * np.sqrt(2 / self.tau_s) * np.random.randn(self.n_neurons)
        return self.s

    def run_full_epoch(self, target_signal, dt=1.0):
        # 1エポックを通して状態をリセットせずに連続実行
        self.reset_state()
        timesteps = len(target_signal)
        outputs = np.zeros((timesteps, self.n_outputs))
        
        z_bar_prev = np.zeros(self.n_neurons)
        z_bar_prev_prev = np.zeros(self.n_neurons)
        psi_prev = np.zeros(self.n_neurons)
        
        for t in range(timesteps):
            # 論文(Fig 1B, 1C)に基づくフェーズの動的切り替え
            if t < 10000:
                phase = "training"
                xi = 1.0
            elif t < 15000:
                phase = "error-driven"
                xi = 1.0
            else:
                phase = "free-running"
                xi = 0.0

            x_t = target_signal[t]
            
            all_z_bar = np.concatenate([self.lif.z_bar, self.alif.z_bar])
            y_t = self.w_out @ all_z_bar
            outputs[t] = y_t
            
            # 予測誤差の計算 (Free-running時も出力記録のため計算はするがネットワークには入力されない)
            d_t = y_t - x_t
            s_t = self.generate_ou_noise(dt)
            
            all_z_bar_bar = np.concatenate([self.lif.z_bar_bar, self.alif.z_bar_bar])
            i_rec = self.w_rec @ all_z_bar_bar
            
            # xi = 0 の場合、外部からのエラー入力(i_in)は遮断される
            i_in = self.w_in @ d_t
            i_fb = self.w_fb @ y_t
            
            total_current = xi * i_in + i_rec + i_fb + self.I_bias + s_t
            
            self.lif.step(total_current[:self.n_lif])
            self.alif.step(total_current[self.n_lif:])
            
            # トレーニングフェーズのみ重みを更新
            if phase == "training":
                c_rd = 1.0 / (50.0 * 125.0)
                L_t = (self.B @ d_t) * c_rd
                
                all_psi = np.concatenate([self.lif.psi, self.alif.psi])
                should_update = self.optimizer.update_traces_and_gradients(
                    self.neurons, z_bar_prev, z_bar_prev_prev, psi_prev, L_t, d_t
                )
                
                if should_update:
                    self.w_rec, self.w_out = self.optimizer.apply_weight_update(self.w_rec, self.w_out)
                
                z_bar_prev_prev = z_bar_prev.copy()
                z_bar_prev = all_z_bar.copy()
                psi_prev = all_psi.copy()
            
        return outputs

# ==========================================
# タスク実行とグラフ描画
# ==========================================
if __name__ == "__main__":
    A, T, phi, c = 0.4, 1000.0, 0.0, 0.0
    # 全フェーズを統合した連続時間軸を作成
    t_all = np.arange(0, 20000, 1)
    
    def sine_wave(t_array):
        return A * np.sin((2 * np.pi / T) * t_array + phi) + c

    target_all = sine_wave(t_all).reshape(-1, 1)
    
    net = PredictiveEPropNet(n_inputs=1, n_lif=100, n_alif=200, n_outputs=1)
    
    epochs = 50
    out_all = None
    
    for epoch in range(epochs):
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        
        # 1つの連続した時系列としてネットワークに投入
        out_all = net.run_full_epoch(target_all)
        
        # フェーズごとにLossを計算
        loss_train = np.mean((out_all[:10000] - target_all[:10000])**2)
        loss_error = np.mean((out_all[10000:15000] - target_all[10000:15000])**2)
        loss_free = np.mean((out_all[15000:] - target_all[15000:])**2)
        
        print(f"  Training Loss: {loss_train:.5f}")
        print(f"  Error-driven Loss: {loss_error:.5f}")
        print(f"  Free-running Loss: {loss_free:.5f}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/predictive_eprop_result_{timestamp}.png"
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # 連続データをフェーズごとにスライスして描画
    axes[0].plot(t_all[:10000], target_all[:10000], label="Target", color="black", linestyle="--")
    axes[0].plot(t_all[:10000], out_all[:10000], label="Output", color="blue", alpha=0.7)
    axes[0].set_title("Training Phase")
    axes[0].set_ylabel("Signal")
    axes[0].legend(loc="upper right")
    
    axes[1].plot(t_all[10000:15000], target_all[10000:15000], label="Target", color="black", linestyle="--")
    axes[1].plot(t_all[10000:15000], out_all[10000:15000], label="Output", color="orange", alpha=0.7)
    axes[1].set_title("Error-driven Phase")
    axes[1].set_ylabel("Signal")
    axes[1].legend(loc="upper right")
    
    axes[2].plot(t_all[15000:], target_all[15000:], label="Target", color="black", linestyle="--")
    axes[2].plot(t_all[15000:], out_all[15000:], label="Output", color="green", alpha=0.7)
    axes[2].set_title("Free-running Phase")
    axes[2].set_xlabel("Time [ms]")
    axes[2].set_ylabel("Signal")
    axes[2].legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\nプロットを画像ファイル '{filename}' として保存しました。")