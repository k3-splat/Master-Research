import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

# ==========================================
# ニューロンモデル
# ==========================================
class BaseNeuronGroup:
    def __init__(self, n_neurons, dt=1.0, tau_mem=20.0, tau_r=50.0, tau_d=125.0, v_th=0.6, gamma_d=0.3, t_ref=5.0):
        self.n = n_neurons
        self.dt = dt
        self.tau_mem = tau_mem
        self.tau_r = tau_r
        self.tau_d = tau_d
        self.v_th = v_th
        self.gamma_d = gamma_d
        self.t_ref = t_ref
        
        self.alpha = np.exp(-dt / tau_mem)
        self.decay_r = np.exp(-dt / tau_r)
        self.decay_d = np.exp(-dt / tau_d)
        
        self.c_rd = 1.0 / (tau_r * tau_d)
        
        self.reset_state()

    def reset_state(self):
        self.v = np.zeros(self.n)
        self.z = np.zeros(self.n)
        self.z_bar = np.zeros(self.n)
        self.z_bar_bar = np.zeros(self.n)
        self.q = np.zeros(self.n) + self.c_rd
        self.psi = np.zeros(self.n)
        self.b = np.full(self.n, self.v_th)
        self.ref_counts = np.zeros(self.n)

    def _update_spikes_and_filters(self, v_next_raw, b_next):
        self.psi = (self.gamma_d / self.v_th) * np.maximum(0, 1.0 - np.abs((v_next_raw - b_next) / self.v_th))
        
        is_currently_refractory = self.ref_counts > 0
        
        self.z = np.where((v_next_raw > b_next) & (~is_currently_refractory), 1.0, 0.0)
        
        v_next = v_next_raw - self.z * b_next
        
        v_next[is_currently_refractory] = self.v[is_currently_refractory]
        
        self.ref_counts = np.maximum(0, self.ref_counts - self.dt)
        self.ref_counts[self.z > 0] = self.t_ref
        
        q_prev = self.q.copy()
        self.q = self.decay_r * self.q + self.z * self.c_rd
        self.z_bar_bar = self.decay_d * self.z_bar_bar + q_prev
        
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
    def __init__(self, n_neurons, tau_alif=2000.0, beta=0.0174, **kwargs):
        super().__init__(n_neurons, **kwargs)
        self.tau_alif = tau_alif
        self.beta = beta
        self.rho = np.exp(-self.dt / tau_alif)
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
    def __init__(self, n_neurons, n_inputs, n_outputs, eta_rec=0.0004, eta_out=0.0004, lambda_reg=2.0, lambda_w=5e-6, t_delay=10, f_star=0.01):
        self.n_neurons = n_neurons
        self.eta_rec = eta_rec
        self.eta_out = eta_out
        self.lambda_reg = lambda_reg
        self.lambda_w = lambda_w
        self.t_delay = t_delay
        self.f_star = f_star
        
        self.e_trace = np.zeros((n_neurons, n_neurons))
        self.eps_b = np.zeros((n_neurons, n_neurons))
        
        self.grad_w_rec = np.zeros((n_neurons, n_neurons))
        self.grad_w_out = np.zeros((n_outputs, n_neurons))
        self.grad_w_rec_accumulated = np.zeros((n_neurons, n_neurons))
        self.grad_w_out_accumulated = np.zeros((n_outputs, n_neurons))
        
        self.spike_buffer = np.zeros(n_neurons)
        self.e_trace_sum = np.zeros((n_neurons, n_neurons))
        self.step_counter = 0
        self.f_bar = np.zeros(n_neurons)

    def reset_traces(self):
        self.e_trace.fill(0)
        self.eps_b.fill(0)
        self.grad_w_rec.fill(0)
        self.grad_w_out.fill(0)
        self.grad_w_rec_accumulated.fill(0)
        self.grad_w_out_accumulated.fill(0)
        self.spike_buffer.fill(0)
        self.e_trace_sum.fill(0)
        self.step_counter = 0
        self.f_bar.fill(0)

    def update_traces_and_gradients(self, neurons, z_bar_bar_curr, z_bar_prev, z_bar_prev_prev, psi_prev, L_t, error_t):
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
        
        self.grad_w_rec_accumulated += L_t[:, None] * self.e_trace
        
        self.grad_w_out_accumulated += np.outer(error_t, z_bar_bar_curr)
        
        all_spikes = np.concatenate([g.z for g in neurons])
        self.spike_buffer += all_spikes
        self.e_trace_sum += self.e_trace
        self.step_counter += 1
        
        if self.step_counter >= self.t_delay:
            self.grad_w_rec = self.grad_w_rec_accumulated.copy()
            self.grad_w_out = self.grad_w_out_accumulated.copy()

            f_bar = self.spike_buffer / self.t_delay
            reg_signal = (self.lambda_reg / self.t_delay) * (f_bar - self.f_star)
            self.grad_w_rec += reg_signal[:, None] * self.e_trace_sum
            
            self.spike_buffer.fill(0)
            self.e_trace_sum.fill(0)
            self.grad_w_rec_accumulated.fill(0)
            self.grad_w_out_accumulated.fill(0)
            self.step_counter = 0
            return True
        return False

    def apply_weight_update(self, w_rec, w_out):
        w_rec -= self.eta_rec * self.grad_w_rec + self.eta_rec * (2 * self.lambda_w) * w_rec
        w_out -= self.eta_out * self.grad_w_out + self.eta_out * (2 * self.lambda_w) * w_out
        
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
        self.g = g
        
        self.lif = LIFGroup(n_lif)
        self.alif = ALIFGroup(n_alif)
        self.neurons = [self.lif, self.alif]
        
        self.w_in = np.random.randn(self.n_neurons, n_inputs) / np.sqrt(g)
        self.w_fb = np.random.randn(self.n_neurons, n_outputs) / np.sqrt(g)
        
        self.density = 0.01
        self.w_rec_mask = (np.random.rand(self.n_neurons, self.n_neurons) < self.density).astype(float)
        np.fill_diagonal(self.w_rec_mask, 0.0)
        
        self.w_rec = (np.random.randn(self.n_neurons, self.n_neurons) / np.sqrt(self.n_neurons)) * self.w_rec_mask

        self.w_out = np.random.randn(n_outputs, self.n_neurons) / np.sqrt(self.n_neurons)
        self.B = np.random.randn(self.n_neurons, n_outputs)
        
        self.I_bias = 0.02
        self.tau_s = 250.0
        self.sigma_s = 1.0
        self.mu_s = 0.0
        self.s = np.zeros(self.n_neurons)
        
        self.optimizer = EPropOptimizer(self.n_neurons, n_inputs, n_outputs) 

        self.rec_z = None
        self.rec_L = None
        self.rec_e = None
        self.rec_f = None 

    def reset_state(self):
        self.lif.reset_state()
        self.alif.reset_state()
        self.optimizer.reset_traces()
        self.s = np.zeros(self.n_neurons)

    def generate_ou_noise(self, dt=1.0):
        decay = np.exp(-dt / self.tau_s)
        # 修正箇所: 論文実装の意図に合わせ、(dt/tau_s)のスケーリングによるアッテネーションを復活
        self.s = self.mu_s + (self.s - self.mu_s) * decay + self.sigma_s * (dt / self.tau_s) * np.random.randn(self.n_neurons)
        return self.s

    def run_full_epoch(self, target_signal, dt=1.0, record=False):
        self.reset_state()
        timesteps = len(target_signal)
        outputs = np.zeros((timesteps, self.n_outputs))
        
        z_bar_prev = np.zeros(self.n_neurons)
        z_bar_prev_prev = np.zeros(self.n_neurons)
        psi_prev = np.zeros(self.n_neurons)
        
        if record:
            self.rec_z = np.zeros((timesteps, self.n_neurons), dtype=bool)
            self.rec_L = np.zeros(timesteps)
            self.rec_e = np.zeros(timesteps)
            self.rec_f = np.zeros(timesteps) 
            
        for t in range(timesteps):
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
            
            all_z_bar_bar = np.concatenate([self.lif.z_bar_bar, self.alif.z_bar_bar])
            all_z_bar = np.concatenate([self.lif.z_bar, self.alif.z_bar])
            
            y_t = self.w_out @ all_z_bar_bar
            outputs[t] = y_t
            
            d_t = y_t - x_t
            s_t = self.generate_ou_noise(dt)
            
            i_rec = self.w_rec @ all_z_bar_bar
            i_in = self.w_in @ d_t
            i_fb = self.w_fb @ y_t
            
            total_current = xi * i_in + i_rec + i_fb + self.I_bias + s_t
            
            self.lif.step(total_current[:self.n_lif])
            self.alif.step(total_current[self.n_lif:])
            
            if phase == "training":
                L_t = (self.B @ d_t) * self.lif.c_rd
                
                all_psi = np.concatenate([self.lif.psi, self.alif.psi])
                
                should_update = self.optimizer.update_traces_and_gradients(
                    self.neurons, all_z_bar, z_bar_prev, z_bar_prev_prev, psi_prev, L_t, d_t
                )
                
                if record:
                    self.rec_z[t] = np.concatenate([self.lif.z, self.alif.z]) > 0
                    self.rec_L[t] = L_t[0].copy()
                    self.rec_e[t] = np.mean(np.abs(self.optimizer.e_trace))
                    self.rec_f[t] = np.mean(self.optimizer.spike_buffer / self.optimizer.t_delay) * 1000
                
                if should_update:
                    self.w_rec, self.w_out = self.optimizer.apply_weight_update(self.w_rec, self.w_out)
                    self.w_rec *= self.w_rec_mask
                
                z_bar_prev_prev = z_bar_prev.copy()
                z_bar_prev = all_z_bar.copy()
                psi_prev = all_psi.copy()
            else:
                if record:
                    self.rec_z[t] = np.concatenate([self.lif.z, self.alif.z]) > 0
                    current_spikes = np.concatenate([self.lif.z, self.alif.z])
                    self.optimizer.spike_buffer += current_spikes
                    self.optimizer.step_counter += 1
                    if self.optimizer.step_counter >= self.optimizer.t_delay:
                        self.optimizer.spike_buffer.fill(0)
                        self.optimizer.step_counter = 0
                    self.rec_f[t] = np.mean(current_spikes) * 1000 
            
        return outputs

# ==========================================
# タスク実行とグラフ描画
# ==========================================
if __name__ == "__main__":
    A, T, phi, c = 0.4, 1000.0, 0.0, 0.0
    
    len_train = 10000
    len_error = 5000
    len_free = 5000
    total_time = len_train + len_error + len_free
    
    t_all = np.arange(0, total_time, 1)
    
    def sine_wave(t_array):
        return A * np.sin((2 * np.pi / T) * t_array + phi) + c

    target_all = sine_wave(t_all).reshape(-1, 1)
    
    net = PredictiveEPropNet(n_inputs=1, n_lif=100, n_alif=200, n_outputs=1)
    
    epochs = 50
    out_all = None
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output", exist_ok=True)
    
    for epoch in range(epochs):
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        
        do_record = (epoch + 1) % 10 == 0
        out_all = net.run_full_epoch(target_all, record=do_record)
        
        loss_train = np.mean((out_all[:len_train] - target_all[:len_train])**2)
        loss_error = np.mean((out_all[len_train:len_train+len_error] - target_all[len_train:len_train+len_error])**2)
        loss_free = np.mean((out_all[len_train+len_error:] - target_all[len_train+len_error:])**2)
        
        print(f"  Training Loss: {loss_train:.5f}")
        print(f"  Error-driven Loss: {loss_error:.5f}")
        print(f"  Free-running Loss: {loss_free:.5f}")
        
        if do_record:
            fig_analysis, axes_analysis = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
            
            spikes = net.rec_z
            y_idx, x_idx = np.where(spikes.T)
            axes_analysis[0].scatter(x_idx, y_idx, s=1, color='black', alpha=0.5)
            axes_analysis[0].set_title(f"Spikes z (Epoch {epoch+1})")
            axes_analysis[0].set_ylabel("Neuron index")
            
            axes_analysis[1].plot(t_all, net.rec_L, color='blue')
            axes_analysis[1].set_title("Learning Signal L (Neuron 0)")
            axes_analysis[1].set_ylabel("L")
            
            axes_analysis[2].plot(t_all, net.rec_e, color='red')
            axes_analysis[2].set_title("Eligibility Trace e (Mean Absolute Value)")
            axes_analysis[2].set_ylabel("e")
            
            axes_analysis[3].plot(t_all, net.rec_f, color='green')
            axes_analysis[3].set_title("Mean Firing Rate")
            axes_analysis[3].set_xlabel("Time [ms]")
            axes_analysis[3].set_ylabel("Hz")
            
            plt.tight_layout()
            analysis_filename = f"output/analysis_epoch_{epoch+1}_{timestamp}.png"
            plt.savefig(analysis_filename)
            plt.close(fig_analysis)
            print(f"  => 内部状態のプロットを '{analysis_filename}' に保存しました。")
    
    img_filename = f"output/predictive_eprop_result_{timestamp}.png"
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    axes[0].plot(t_all[:len_train], target_all[:len_train], label="Target", color="black", linestyle="--")
    axes[0].plot(t_all[:len_train], out_all[:len_train], label="Output", color="blue", alpha=0.7)
    axes[0].set_title("Training Phase")
    axes[0].set_ylabel("Signal")
    axes[0].legend(loc="upper right")
    
    axes[1].plot(t_all[len_train:len_train+len_error], target_all[len_train:len_train+len_error], label="Target", color="black", linestyle="--")
    axes[1].plot(t_all[len_train:len_train+len_error], out_all[len_train:len_train+len_error], label="Output", color="orange", alpha=0.7)
    axes[1].set_title("Error-driven Phase")
    axes[1].set_ylabel("Signal")
    axes[1].legend(loc="upper right")
    
    axes[2].plot(t_all[len_train+len_error:], target_all[len_train+len_error:], label="Target", color="black", linestyle="--")
    axes[2].plot(t_all[len_train+len_error:], out_all[len_train+len_error:], label="Output", color="green", alpha=0.7)
    axes[2].set_title("Free-running Phase")
    axes[2].set_xlabel("Time [ms]")
    axes[2].set_ylabel("Signal")
    axes[2].legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(img_filename)
    print(f"\n推論結果のプロットを画像ファイル '{img_filename}' として保存しました。")
    
    param_filename = f"output/predictive_eprop_params_{timestamp}.txt"
    
    actual_params = {
        "training phase (ms)": len_train,
        "error-driven phase (ms)": len_error,
        "free-running phase (ms)": len_free,
        "eta_rec": net.optimizer.eta_rec,
        "eta_out" : net.optimizer.eta_out,
        "lambda_W": net.optimizer.lambda_w,
        "lambda_reg": net.optimizer.lambda_reg,
        "gamma_d": net.lif.gamma_d,
        "time step (dt)": net.lif.dt,
        "t_delay": net.optimizer.t_delay,
        "f_star": net.optimizer.f_star,
        "A": A,
        "T": T,
        "phi (φ)": phi,
        "c": c,
        "N_LIF": net.n_lif,
        "N_ALIF": net.n_alif,
        "sparsity": 1.0 - net.density,
        "tau_s": net.tau_s,
        "sigma_s": net.sigma_s,
        "mu_s": net.mu_s,
        "g": net.g,
        "v_th": net.lif.v_th,
        "I_bias": net.I_bias,
        "tau_mem": net.lif.tau_mem,
        "tau_alif": net.alif.tau_alif,
        "tau_r": net.lif.tau_r,
        "tau_d": net.lif.tau_d,
        "t_ref": net.lif.t_ref
    }

    with open(param_filename, "w", encoding="utf-8") as f:
        f.write("スクリプト内で使用されたハイパーパラメータ一覧\n")
        f.write("=" * 60 + "\n")
        for key, value in actual_params.items():
            f.write(f"{key}: {value}\n")
            
    print(f"スクリプトで使用されたパラメータ数値をテキストファイル '{param_filename}' として保存しました。")