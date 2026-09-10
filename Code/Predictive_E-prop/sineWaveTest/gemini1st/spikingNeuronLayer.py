import torch
import torch.nn as nn
import datetime

class SpikingNeuronLayer(nn.Module):
    """
    Predictive E-prop実験用のLIF/ALIFニューロン層および順伝播モジュール
    """
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
        
        # ネットワークサイズと定数の初期化
        self.n_lif = n_lif
        self.n_alif = n_alif
        self.n_total = n_lif + n_alif
        self.dt = dt
        
        self.v_th = v_th
        self.i_bias = i_bias
        
        # 減衰定数の計算
        self.alpha = torch.exp(torch.tensor(-dt / tau_mem))
        self.rho = torch.exp(torch.tensor(-dt / tau_alif))
        self.decay_r = torch.exp(torch.tensor(-dt / tau_r))
        self.decay_d = torch.exp(torch.tensor(-dt / tau_d))
        
        self.filter_c = 1.0 / (tau_r * tau_d)
        
        # ALIFニューロン用の適応閾値係数 (LIFは0、ALIFは定数)
        self.beta = torch.zeros(self.n_total)
        self.beta[n_lif:] = beta_alif
        
        # OUノイズ用パラメータ
        self.tau_s = tau_s
        self.sigma_s = sigma_s
        self.ou_decay = 1.0 - (dt / tau_s)
        self.ou_scale = sigma_s * torch.sqrt(torch.tensor(2.0 * dt / tau_s))
        
        # --- 重み行列の初期化 ---
        self.W_in = nn.Parameter(torch.randn(self.n_total, 1) / torch.sqrt(torch.tensor(g)), requires_grad=False)
        self.W_fb = nn.Parameter(torch.randn(self.n_total, 1) / torch.sqrt(torch.tensor(g)), requires_grad=False)
        
        W_rec_init = torch.randn(self.n_total, self.n_total) / torch.sqrt(torch.tensor(float(self.n_total)))
        mask = (torch.rand(self.n_total, self.n_total) > sparsity).float()
        self.W_rec = nn.Parameter(W_rec_init * mask, requires_grad=False) 
        
        self.W_out = nn.Parameter(torch.randn(1, self.n_total) / torch.sqrt(torch.tensor(float(self.n_total))), requires_grad=False)
        
        self.reset_states()

    def reset_states(self):
        """内部状態をゼロにリセットする"""
        self.v = torch.zeros(self.n_total)
        self.a = torch.zeros(self.n_total)
        self.z = torch.zeros(self.n_total)
        
        self.z_bar = torch.zeros(self.n_total)
        self.z_bar_bar = torch.zeros(self.n_total)
        self.q = torch.ones(self.n_total) * self.filter_c
        
        self.s = torch.zeros(self.n_total)
        self.refractory_counter = torch.zeros(self.n_total)

    def forward_step(self, d_t, xi=1.0):
        """1タイムステップ分のネットワーク状態を更新する"""
        y_t_prev = torch.matmul(self.W_out, self.z_bar)
        
        # 1. OUノイズの更新
        noise = torch.randn(self.n_total)
        self.s = self.ou_decay * self.s + self.ou_scale * noise
        
        # 2. 適応閾値の更新
        self.a = self.rho * self.a + self.z
        b = self.v_th + self.beta * self.a
        
        # 3. 膜電位の更新
        # リカレント入力を z_bar から z_bar_bar に修正
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
        
        # 4. 発火判定
        self.z = (self.v >= b).float()
        self.refractory_counter[self.z == 1.0] = 5.0
        
        # 5. スパイクのフィルター処理（修正済み）
        q_next = self.decay_r * self.q + self.filter_c * self.z
        self.z_bar_bar = self.decay_d * self.z_bar_bar + self.q
        self.q = q_next
        self.z_bar = self.alpha * self.z_bar + self.z_bar_bar
        
        # 6. 予測出力の計算
        y_t = torch.matmul(self.W_out, self.z_bar)
        
        return y_t, self.z

# ==========================================
# テスト実行コード (単体テスト用)
# ==========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    network = SpikingNeuronLayer()
    
    time_steps = 1000
    output_history = []
    spike_history = []
    
    for t in range(time_steps):
        dummy_d_t = torch.tensor([0.1])
        y_t, z_t = network.forward_step(dummy_d_t, xi=1.0)
        
        output_history.append(y_t.item())
        spike_history.append(z_t.clone())

    spike_history = torch.stack(spike_history)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    spikes = spike_history.nonzero()
    ax1.scatter(spikes[:, 0].numpy(), spikes[:, 1].numpy(), s=1, c='black')
    ax1.set_title("Spike Raster Plot (Untrained - Fixed)")
    ax1.set_ylabel("Neuron Index")
    ax1.axhline(100, color='red', linestyle='--', linewidth=1, label='LIF / ALIF boundary')
    ax1.legend(loc="upper right")
    
    ax2.plot(output_history, color='blue')
    ax2.set_title("Network Output $y^t$")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Signal")
    
    plt.tight_layout()
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"spiking_layer_fixed_{current_time}.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    
    plt.close()