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
                 beta_alif=1.0): # beta_alifはヒューリスティックに決定される定数
        
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
        # W_in, W_fb: N(0,1)/sqrt(g) -> 学習中は固定
        self.W_in = nn.Parameter(torch.randn(self.n_total, 1) / torch.sqrt(torch.tensor(g)), requires_grad=False)
        self.W_fb = nn.Parameter(torch.randn(self.n_total, 1) / torch.sqrt(torch.tensor(g)), requires_grad=False)
        
        # W_rec, W_out: N(0,1)/sqrt(N_total) -> e-propで学習 (ここでは便宜上Parameterとして定義)
        W_rec_init = torch.randn(self.n_total, self.n_total) / torch.sqrt(torch.tensor(float(self.n_total)))
        # スパース性の適用 (例: sparsity=0.99 なら 99% の結合を 0 にする)
        mask = (torch.rand(self.n_total, self.n_total) > sparsity).float()
        self.W_rec = nn.Parameter(W_rec_init * mask, requires_grad=False) 
        
        self.W_out = nn.Parameter(torch.randn(1, self.n_total) / torch.sqrt(torch.tensor(float(self.n_total))), requires_grad=False)
        
        # 状態変数のリセット
        self.reset_states()

    def reset_states(self):
        """内部状態（膜電位、フィルター値、ノイズなど）をゼロにリセットする"""
        self.v = torch.zeros(self.n_total)
        self.a = torch.zeros(self.n_total)
        self.z = torch.zeros(self.n_total)
        
        self.z_bar = torch.zeros(self.n_total)      # z bar (Low-pass filtered spike train)
        self.z_bar_bar = torch.zeros(self.n_total)  # z double bar
        self.q = torch.ones(self.n_total) * self.filter_c
        
        self.s = torch.zeros(self.n_total)          # OU Noise
        self.refractory_counter = torch.zeros(self.n_total) # 不応期管理 (5ms)

    def forward_step(self, d_t, xi=1.0):
        """
        1タイムステップ分のネットワーク状態を更新する
        
        Args:
            d_t (torch.Tensor): 予測誤差入力 (形状: [1])
            xi (float): エラー駆動フェーズのスイッチ (1.0 or 0.0)
            
        Returns:
            y_t (torch.Tensor): ネットワークの予測出力 (形状: [1])
            z_t (torch.Tensor): スパイク状態 (形状: [n_total])
        """
        # 前時刻の予測値 y_{t-1} を計算（フィードバック用）
        y_t_prev = torch.matmul(self.W_out, self.z_bar)
        
        # 1. OUノイズの更新
        noise = torch.randn(self.n_total)
        self.s = self.ou_decay * self.s + self.ou_scale * noise
        
        # 2. 適応閾値 b_i^t の更新 (a_i^t = rho * a_i^{t-1} + z_i^{t-1})
        self.a = self.rho * self.a + self.z
        b = self.v_th + self.beta * self.a
        
        # 3. 膜電位 v_i^{t+1} の更新
        # 入力: 誤差入力 + リカレント入力 + フィードバック入力 + バイアス + ノイズ
        input_current = (xi * torch.matmul(self.W_in, d_t) + 
                         torch.matmul(self.W_rec, self.z_bar) + 
                         torch.matmul(self.W_fb, y_t_prev).squeeze() + 
                         self.i_bias + 
                         self.s)
        
        # 再分極 (Repolarization): スパイク後に閾値分下げる
        repolarization = self.z * b
        
        # 電位の更新
        v_next = self.alpha * self.v + input_current - repolarization
        
        # 不応期 (5ms) の適用: カウンターが0より大きい場合は電位を固定
        is_refractory = self.refractory_counter > 0
        v_next = torch.where(is_refractory, self.v, v_next)
        
        self.v = v_next
        
        # 不応期カウンターの更新 (0以下にはしない)
        self.refractory_counter = torch.clamp(self.refractory_counter - self.dt, min=0.0)
        
        # 4. 発火判定 (Heaviside step-function)
        self.z = (self.v >= b).float()
        
        # スパイクが発生したニューロンの不応期カウンターを5msにセット
        self.refractory_counter[self.z == 1.0] = 5.0
        
        # 5. スパイクの二重指数フィルター処理
        # q_i^t = exp(-dt/tau_r) * q_i^{t-1} + (1 / tau_r*tau_d) * z_i^t
        q_next = self.decay_r * self.q + self.filter_c * self.z
        
        # z_bar_bar_i^t = exp(-dt/tau_d) * z_bar_i^{t-1} + q_i^{t-1}
        self.z_bar_bar = self.decay_d * self.z_bar + self.q
        self.q = q_next
        
        # z_bar_i^t = alpha * z_bar_i^{t-1} + z_bar_bar_i^t
        self.z_bar = self.alpha * self.z_bar + self.z_bar_bar
        
        # 6. 新しい予測値 y_t の計算
        y_t = torch.matmul(self.W_out, self.z_bar)
        
        return y_t, self.z

# ==========================================
# テスト実行コード (単体テスト用)
# ==========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ネットワークの初期化
    network = SpikingNeuronLayer()
    
    # テスト用のシミュレーション（1000ステップ＝1000ms）
    time_steps = 1000
    output_history = []
    spike_history = []
    
    for t in range(time_steps):
        # テストとして、外部誤差 d_t にダミーの微小信号を与える
        dummy_d_t = torch.tensor([0.1])
        y_t, z_t = network.forward_step(dummy_d_t, xi=1.0)
        
        output_history.append(y_t.item())
        spike_history.append(z_t.clone())

    # スパイク履歴のテンソル化
    spike_history = torch.stack(spike_history) # 形状: [1000, 300]
    
    # 描画処理: スパイクラスタプロットとネットワーク出力
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # ラスタプロットの描画
    spikes = spike_history.nonzero()
    ax1.scatter(spikes[:, 0].numpy(), spikes[:, 1].numpy(), s=1, c='black')
    ax1.set_title("Spike Raster Plot (Untrained)")
    ax1.set_ylabel("Neuron Index")
    ax1.axhline(100, color='red', linestyle='--', linewidth=1, label='LIF / ALIF boundary')
    ax1.legend(loc="upper right")
    
    # 出力信号の描画
    ax2.plot(output_history, color='blue')
    ax2.set_title("Network Output $y^t$")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Signal")
    
    plt.tight_layout()
    
    # 日時を取得してユニークなファイル名を生成し、画像を保存
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"spiking_layer_test_{current_time}.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    
    # リソース解放
    plt.close()