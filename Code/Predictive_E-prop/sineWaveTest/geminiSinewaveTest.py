import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================================
# 1. ハイパーパラメータ設定 (Table SI 準拠)
# ==========================================
params = {
    'dt': 1.0,               # タイムステップ (ms)
    'N_LIF': 100,            # LIFニューロン数
    'N_ALIF': 200,           # ALIFニューロン数
    'tau_mem': 20.0,         # 膜電位時定数
    'tau_alif': 2000.0,      # 適応閾値時定数
    'v_th': 0.6,             # 基本発火閾値
    'I_bias': 0.02,          # バイアス電流
    'tau_r': 50.0,           # フィルタ立ち上がり時定数
    'tau_d': 125.0,          # フィルタ減衰時定数
    'eta': 0.0004,           # 学習率
    'g': 150.0,              # 入力・フィードバックゲイン
    'gamma_d': 0.3,          # サロゲート勾配（テント関数）の高さ
    'beta': 1.0,             # ALIFニューロンの適応係数
    't_delay': 10,           # 重み更新・恒常性レギュラレーションの間隔 (ms)
    'f_target': 10.0 / 1000, # 目標発火率 (10 Hz = 0.01 spikes/ms)
    'lambda_reg': 2.0,       # 発火率正則化係数
    'lambda_w': 5e-06,       # 重み減衰係数
    'tau_s': 250.0,          # OUノイズ時定数
    'sigma_s': 1.0,          # OUノイズ標準偏差
    'A': 0.4,                # Sine波振幅
    'T': 1000.0,             # Sine波周期 (ms)
}

# ==========================================
# 2. モデル定義 (Predictive E-prop RSNN)
# ==========================================
class PredictiveEPropRSNN:
    def __init__(self, p):
        self.p = p
        self.N_rec = p['N_LIF'] + p['N_ALIF']
        self.N_out = 1
        
        # 減衰定数
        self.alpha = np.exp(-self.p['dt'] / self.p['tau_mem'])
        self.rho = np.exp(-self.p['dt'] / self.p['tau_alif'])
        self.decay_d = np.exp(-self.p['dt'] / self.p['tau_d'])
        self.decay_r = np.exp(-self.p['dt'] / self.p['tau_r'])

        # ALIFフラグ (前半100個がLIF、後半200個がALIF)
        self.is_alif = torch.zeros(self.N_rec)
        self.is_alif[self.p['N_LIF']:] = 1.0

        # --- 重みの初期化 ---
        self.W_in = torch.randn(self.N_rec, self.N_out) / np.sqrt(self.p['g'])
        self.W_fb = torch.randn(self.N_rec, self.N_out) / np.sqrt(self.p['g'])
        self.B = torch.randn(self.N_rec, self.N_out) 
        
        self.W_rec = torch.randn(self.N_rec, self.N_rec) / np.sqrt(self.N_rec)
        mask = (torch.rand(self.N_rec, self.N_rec) > 0.99).float()
        self.W_rec *= mask
        self.W_out = torch.randn(self.N_out, self.N_rec) / np.sqrt(self.N_rec)

        self.reset_states()

    def reset_states(self):
        # ネットワーク状態の初期化
        self.v = torch.zeros(self.N_rec)
        self.a = torch.zeros(self.N_rec)
        self.z = torch.zeros(self.N_rec)
        
        self.q = torch.ones(self.N_rec) / (self.p['tau_r'] * self.p['tau_d'])
        self.z_bar = torch.zeros(self.N_rec)
        
        self.s = torch.zeros(self.N_rec)
        
        self.epsilon_v = torch.zeros(self.N_rec, self.N_rec)
        self.epsilon_b = torch.zeros(self.N_rec, self.N_rec)
        self.e_trace = torch.zeros(self.N_rec, self.N_rec)
        
        self.refractory_timer = torch.zeros(self.N_rec)
        
        self.dW_rec = torch.zeros_like(self.W_rec)
        self.dW_out = torch.zeros_like(self.W_out)

    def step(self, d_t, y_prev, xi, update_weights=False):
        # 1. OUノイズの更新
        noise = torch.randn(self.N_rec) * self.p['sigma_s']
        self.s = self.s * np.exp(-self.p['dt']/self.p['tau_s']) + noise * np.sqrt(1 - np.exp(-2*self.p['dt']/self.p['tau_s']))

        # 2. 膜電位と閾値の更新
        b = self.p['v_th'] + self.p['beta'] * self.is_alif * self.a
        active = (self.refractory_timer <= 0).float()
        dv = (self.W_in @ d_t * xi) + (self.W_rec @ self.z_bar) + (self.W_fb @ y_prev) + self.p['I_bias'] + self.s
        self.v = self.alpha * self.v + dv * active
        
        surpass = self.v - b
        self.z = (surpass > 0).float() * active
        
        self.v = self.v - self.z * b
        self.refractory_timer[self.z > 0] = 5.0
        self.refractory_timer -= self.p['dt']

        self.a = self.rho * self.a + self.z

        # 3. フィルタの更新
        self.q = self.decay_r * self.q + (1.0 / (self.p['tau_r'] * self.p['tau_d'])) * self.z
        self.z_bar = self.decay_d * self.z_bar + self.q

        # 4. 出力予測
        y = self.W_out @ self.z_bar

        # E-prop学習計算
        if update_weights:
            psi = (1.0 / self.p['v_th']) * self.p['gamma_d'] * torch.clamp(1.0 - torch.abs(surpass / self.p['v_th']), min=0.0)
            self.epsilon_v = self.alpha * self.epsilon_v + self.z_bar.unsqueeze(0)
            
            psi_prev = psi.unsqueeze(1)
            self.epsilon_b = psi_prev * self.z_bar.unsqueeze(0) + (self.rho - self.p['beta'] * psi_prev) * self.epsilon_b
            
            self.e_trace = psi.unsqueeze(1) * (self.z_bar.unsqueeze(0) - self.p['beta'] * self.is_alif.unsqueeze(1) * self.epsilon_b)
            
            L = self.B @ d_t * (1.0 / (self.p['tau_r'] * self.p['tau_d']))
            
            self.dW_rec += self.p['eta'] * L.unsqueeze(1) * self.e_trace
            self.dW_out += self.p['eta'] * d_t.unsqueeze(1) * self.z_bar.unsqueeze(0)

        return y, self.z

# ==========================================
# 3. 実験ループ (50 Epochs / 3 Phases)
# ==========================================
def run_experiment_full(params):
    model = PredictiveEPropRSNN(params)
    
    T_train = 10000
    T_error = 5000
    T_free = 5000
    total_steps = T_train + T_error + T_free
    
    t_total = np.arange(0, total_steps, params['dt'])
    target_signal = params['A'] * np.sin((2 * np.pi / params['T']) * t_total)
    
    epochs = 50 
    
    print("シミュレーションを開始します。完了まで時間がかかる場合があります...")
    for epoch in range(epochs):
        model.reset_states()
        y = torch.zeros(1)
        spike_history = []
        outputs = []
        spikes_out = []
        
        for t_step in range(total_steps):
            # フェーズの判定
            if t_step < T_train:
                phase = "Training"
                xi = 1.0
                update_weights = True
            elif t_step < T_train + T_error:
                phase = "Error-Driven"
                xi = 1.0
                update_weights = False
            else:
                phase = "Free-Running"
                xi = 0.0
                update_weights = False
                
            target_t = torch.tensor([target_signal[t_step]], dtype=torch.float32)
            d_t = y - target_t if phase != "Free-Running" else torch.zeros(1)
            
            # 発火率の計算と恒常性正則化の勾配ペナルティ
            if update_weights and t_step % params['t_delay'] == 0:
                model.dW_rec = torch.zeros_like(model.W_rec)
                model.dW_out = torch.zeros_like(model.W_out)
                if len(spike_history) >= params['t_delay']:
                    recent_spikes = torch.tensor(np.array(spike_history[-params['t_delay']:])).mean(dim=0)
                    f_error = recent_spikes - params['f_target']
                    model.dW_rec -= params['lambda_reg'] * f_error.unsqueeze(1) * model.e_trace
                    
            y, z = model.step(d_t, y, xi, update_weights=update_weights)
            
            spike_history.append(z.numpy())
            
            # 最終エポックのデータのみ保持
            if epoch == epochs - 1:
                outputs.append(y.item())
                spikes_out.append(z.numpy())
            
            # 重みの適用とWeight Decay
            if update_weights and (t_step + 1) % params['t_delay'] == 0:
                model.dW_rec -= params['lambda_w'] * model.W_rec
                model.dW_out -= params['lambda_w'] * model.W_out
                model.W_rec += model.dW_rec
                model.W_out += model.dW_out
                
        print(f"Epoch {epoch+1}/{epochs} completed.")
        
    return target_signal, np.array(outputs), np.array(spikes_out)

if __name__ == "__main__":
    target, output, spikes = run_experiment_full(params)
    
    # プロットの作成
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(target, label="Target (Sine)", linestyle="--", color='black')
    plt.plot(output, label="Network Output", color='red', alpha=0.7)
    plt.axvline(x=10000, color='blue', linestyle=':', label='Start Error-Driven')
    plt.axvline(x=15000, color='green', linestyle=':', label='Start Free-Running')
    plt.title("Predictive E-prop: Time-Series Generation Task (Epoch 50)")
    plt.ylabel("Signal")
    plt.legend(loc='upper right')
    
    plt.subplot(2, 1, 2)
    spike_indices, spike_times = np.where(spikes[:, :50].T > 0)
    plt.scatter(spike_times, spike_indices, s=1, color='black', alpha=0.5)
    plt.axvline(x=10000, color='blue', linestyle=':')
    plt.axvline(x=15000, color='green', linestyle=':')
    plt.ylabel("Neuron ID (0-49)")
    plt.xlabel("Time [ms]")
    
    plt.tight_layout()
    
    # ==========================================
    # 画像の保存処理
    # ==========================================
    # 現在の日時を取得してファイル名を作成 (YYYYMMDD_HHMMSS)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictive_eprop_result_{current_time}.png"
    
    # 画像を保存 (高解像度 300 dpi)
    plt.savefig(filename, dpi=300)
    print(f"\nプロット画像を保存しました: {filename}")
    
    # 画像を表示
    plt.show()