import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import datetime
import os

# ==========================================
# 1. データ生成モジュール
# ==========================================
class SineWaveGenerator:
    def __init__(self, amplitude=0.4, period=1000.0, phase=0.0, offset=0.0, dt=1.0):
        self.amplitude = amplitude
        self.period = period
        self.phase = phase
        self.offset = offset
        self.dt = dt

    def generate(self, total_time):
        time_steps = int(total_time / self.dt)
        t = torch.arange(0, time_steps, dtype=torch.float32) * self.dt
        signal = self.amplitude * torch.sin((2 * torch.pi / self.period) * t + self.phase) + self.offset
        return signal.unsqueeze(1)


# ==========================================
# 2. ネットワーク層モジュール
# ==========================================
class SpikingNeuronLayer(nn.Module):
    def __init__(self, 
                 n_lif=100, 
                 n_alif=200, 
                 dt=1.0, 
                 tau_mem=20.0, 
                 tau_alif=2000.0,
                 tau_r=50.0, 
                 tau_d=125.0,
                 v_th=0.6, 
                 i_bias=0.02, 
                 g=150.0, 
                 sparsity=0.99,
                 tau_s=250.0,
                 sigma_s=1.0,
                 beta_alif=1.0):
        
        super(SpikingNeuronLayer, self).__init__()
        
        self.n_lif = n_lif
        self.n_alif = n_alif
        self.n_total = n_lif + n_alif
        self.dt = dt
        
        self.v_th = v_th
        self.i_bias = i_bias
        
        self.alpha = torch.exp(torch.tensor(-dt / tau_mem))
        self.rho = torch.exp(torch.tensor(-dt / tau_alif))
        self.decay_r = torch.exp(torch.tensor(-dt / tau_r))
        self.decay_d = torch.exp(torch.tensor(-dt / tau_d))
        
        self.filter_c = 1.0 / (tau_r * tau_d)
        
        self.beta = torch.zeros(self.n_total)
        self.beta[n_lif:] = beta_alif
        
        self.tau_s = tau_s
        self.sigma_s = sigma_s
        self.ou_decay = 1.0 - (dt / tau_s)
        self.ou_scale = sigma_s * torch.sqrt(torch.tensor(2.0 * dt / tau_s))
        
        self.W_in = nn.Parameter(torch.randn(self.n_total, 1) / torch.sqrt(torch.tensor(g)), requires_grad=False)
        self.W_fb = nn.Parameter(torch.randn(self.n_total, 1) / torch.sqrt(torch.tensor(g)), requires_grad=False)
        
        W_rec_init = torch.randn(self.n_total, self.n_total) / torch.sqrt(torch.tensor(float(self.n_total)))
        mask = (torch.rand(self.n_total, self.n_total) > sparsity).float()
        self.W_rec = nn.Parameter(W_rec_init * mask, requires_grad=False) 
        
        self.W_out = nn.Parameter(torch.randn(1, self.n_total) / torch.sqrt(torch.tensor(float(self.n_total))), requires_grad=False)
        
        self.reset_states()

    def reset_states(self):
        self.v = torch.zeros(self.n_total)
        self.a = torch.zeros(self.n_total)
        self.z = torch.zeros(self.n_total)
        
        self.z_bar = torch.zeros(self.n_total)
        self.z_bar_bar = torch.zeros(self.n_total)
        self.q = torch.ones(self.n_total) * self.filter_c
        
        self.s = torch.zeros(self.n_total)
        self.refractory_counter = torch.zeros(self.n_total)

    def forward_step(self, d_t, xi=1.0):
        y_t_prev = torch.matmul(self.W_out, self.z_bar)
        
        noise = torch.randn(self.n_total)
        self.s = self.ou_decay * self.s + self.ou_scale * noise
        
        self.a = self.rho * self.a + self.z
        b = self.v_th + self.beta * self.a
        
        input_current = (xi * torch.matmul(self.W_in, d_t) + 
                         torch.matmul(self.W_rec, self.z_bar_bar) + 
                         torch.matmul(self.W_fb, y_t_prev).squeeze() + 
                         self.i_bias + 
                         self.s)
        
        repolarization = self.z * b
        v_next = self.alpha * self.v + input_current - repolarization
        
        is_refractory = self.refractory_counter > 0
        v_next = torch.where(is_refractory, self.v, v_next)
        
        self.v = v_next
        self.refractory_counter = torch.clamp(self.refractory_counter - self.dt, min=0.0)
        
        self.z = (self.v >= b).float()
        self.refractory_counter[self.z == 1.0] = 5.0
        
        q_next = self.decay_r * self.q + self.filter_c * self.z
        self.z_bar_bar = self.decay_d * self.z_bar_bar + self.q
        self.q = q_next
        self.z_bar = self.alpha * self.z_bar + self.z_bar_bar
        
        y_t = torch.matmul(self.W_out, self.z_bar)
        return y_t, self.z, b


# ==========================================
# 3. 学習アルゴリズムモジュール
# ==========================================
class EPropOptimizer:
    def __init__(self, 
                 network, 
                 lr=0.0004, 
                 gamma_d=0.3, 
                 lambda_reg=2.0, 
                 lambda_w=5e-6, 
                 f_target=0.01,
                 t_delay=10):
        
        self.network = network
        self.lr = lr
        self.gamma_d = gamma_d
        self.lambda_reg = lambda_reg
        self.lambda_w = lambda_w
        self.f_target = f_target
        self.t_delay = t_delay
        self.n_total = network.n_total
        
        self.B = torch.randn(self.n_total, 1)
        self.reset_states()

    def reset_states(self):
        self.eps_b = torch.zeros(self.n_total, self.n_total)
        self.psi_prev = torch.zeros(self.n_total)
        self.z_bar_prev1 = torch.zeros(self.n_total)
        self.z_bar_prev2 = torch.zeros(self.n_total)
        
        self.dW_rec = torch.zeros(self.n_total, self.n_total)
        self.dW_out = torch.zeros(1, self.n_total)
        
        self.spike_count = torch.zeros(self.n_total)
        self.step_counter = 0

    def step(self, x_t, y_t, v_t, b_t, z_t, z_bar_t, is_training=True):
        d_t = y_t - x_t
        v_th = self.network.v_th
        abs_diff = torch.abs((v_t - b_t) / v_th)
        psi_t = (self.gamma_d / v_th) * torch.clamp(1.0 - abs_diff, min=0.0)
        
        if is_training:
            term1 = torch.outer(self.psi_prev, self.z_bar_prev2)
            beta_matrix = self.network.beta.unsqueeze(1)
            term2_coef = self.network.rho - (beta_matrix * self.psi_prev.unsqueeze(1))
            self.eps_b = term1 + term2_coef * self.eps_b
            
            z_bar_prev1_matrix = self.z_bar_prev1.unsqueeze(0).repeat(self.n_total, 1)
            e_trace = psi_t.unsqueeze(1) * (z_bar_prev1_matrix - beta_matrix * self.eps_b)
            
            L_t = self.B * d_t * self.network.filter_c
            
            self.dW_rec += self.lr * L_t * e_trace
            self.dW_out += self.lr * d_t * z_bar_t.unsqueeze(0)
            
            self.spike_count += z_t
            self.step_counter += 1
            
            if self.step_counter >= self.t_delay:
                self.apply_updates_and_regularization()
                self.spike_count.zero_()
                self.step_counter = 0

        self.psi_prev = psi_t.clone()
        self.z_bar_prev2 = self.z_bar_prev1.clone()
        self.z_bar_prev1 = z_bar_t.clone()

    def apply_updates_and_regularization(self):
        with torch.no_grad():
            self.network.W_rec += self.dW_rec
            self.network.W_out += self.dW_out
            
            self.network.W_rec -= self.lambda_w * self.network.W_rec
            self.network.W_out -= self.lambda_w * self.network.W_out
            
            f_bar = self.spike_count / self.t_delay
            f_error = f_bar - self.f_target
            reg_penalty = self.lambda_reg * f_error.unsqueeze(1).repeat(1, self.n_total)
            self.network.W_rec -= self.lr * reg_penalty
            
            self.dW_rec.zero_()
            self.dW_out.zero_()


# ==========================================
# 4. 実験制御モジュール
# ==========================================
class ExperimentController:
    def __init__(self, generator, network, optimizer):
        self.generator = generator
        self.network = network
        self.optimizer = optimizer
        
        # 各フェーズの時間設定 (ms)
        self.t_train = 10000
        self.t_error = 5000
        self.t_free = 5000
        self.total_time = self.t_train + self.t_error + self.t_free
        
        # ターゲット信号の事前生成
        self.target_signal = self.generator.generate(self.total_time)

    def run_epoch(self):
        self.network.reset_states()
        self.optimizer.reset_states()
        
        outputs = []
        spikes = []
        
        print("Starting Epoch...")
        
        for t in range(int(self.total_time)):
            x_t = self.target_signal[t]
            
            # フェーズ制御
            if t < self.t_train:
                # Training phase: 誤差入力あり, 重み更新あり
                xi = 1.0
                is_training = True
            elif t < self.t_train + self.t_error:
                # Error-driven phase: 誤差入力あり, 重み更新なし
                xi = 1.0
                is_training = False
            else:
                # Free-running phase: 誤差入力なし, 重み更新なし
                xi = 0.0
                is_training = False
            
            # 予測誤差の計算（t=0では出力は0とみなす）
            y_t_prev = outputs[-1] if t > 0 else torch.tensor([0.0])
            d_t = y_t_prev - x_t if xi == 1.0 else torch.zeros(1)
            
            # 順伝播
            y_t, z_t, b_t = self.network.forward_step(d_t, xi=xi)
            
            # 学習ステップ
            self.optimizer.step(
                x_t=x_t, 
                y_t=y_t, 
                v_t=self.network.v, 
                b_t=b_t, 
                z_t=z_t, 
                z_bar_t=self.network.z_bar, 
                is_training=is_training
            )
            
            outputs.append(y_t.clone())
            
            # メモリ節約のため、スパイクデータは間引いて保存（全保存すると重くなるため）
            if t % 10 == 0:
                spikes.append((t, z_t.nonzero().squeeze(1).numpy()))
                
            if (t + 1) % 5000 == 0:
                print(f"  Processed {t + 1} / {self.total_time} ms")
                
        return torch.stack(outputs).squeeze(), self.target_signal.squeeze(), spikes

    def plot_results(self, outputs, targets, spikes):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # ネットワーク出力 vs ターゲット
        t_axis = torch.arange(self.total_time).numpy()
        ax1.plot(t_axis, targets.numpy(), label='Target', color='black', alpha=0.6)
        ax1.plot(t_axis, outputs.numpy(), label='Output', color='red', alpha=0.8)
        
        # フェーズの背景色設定
        ax1.axvspan(0, self.t_train, color='green', alpha=0.1, label='Training')
        ax1.axvspan(self.t_train, self.t_train + self.t_error, color='cyan', alpha=0.1, label='Error-driven')
        ax1.axvspan(self.t_train + self.t_error, self.total_time, color='purple', alpha=0.1, label='Free-running')
        
        ax1.set_title("Network Output vs Target Signal")
        ax1.set_ylabel("Signal")
        ax1.legend(loc='upper right')
        
        # スパイクラスタプロット
        for t, active_neurons in spikes:
            if len(active_neurons) > 0:
                ax2.scatter([t] * len(active_neurons), active_neurons, s=1, c='black')
        
        ax2.axvspan(0, self.t_train, color='green', alpha=0.1)
        ax2.axvspan(self.t_train, self.t_train + self.t_error, color='cyan', alpha=0.1)
        ax2.axvspan(self.t_train + self.t_error, self.total_time, color='purple', alpha=0.1)
        
        ax2.axhline(100, color='red', linestyle='--', linewidth=1, label='LIF / ALIF boundary')
        ax2.set_title("Spike Raster Plot (Subsampled 1/10)")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Neuron Index")
        
        plt.tight_layout()
        
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictive_eprop_epoch_{current_time}.png"
        plt.savefig(filename)
        print(f"\nPlot saved as {filename}")
        plt.close()


# ==========================================
# 実行エントリーポイント
# ==========================================
if __name__ == "__main__":
    generator = SineWaveGenerator()
    network = SpikingNeuronLayer()
    optimizer = EPropOptimizer(network)
    
    controller = ExperimentController(generator, network, optimizer)
    
    # 1エポック分のシミュレーションを実行
    outputs, targets, spikes = controller.run_epoch()
    
    # 結果のプロットと保存
    controller.plot_results(outputs, targets, spikes)