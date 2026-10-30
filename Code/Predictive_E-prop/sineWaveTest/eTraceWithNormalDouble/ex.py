import torch
import matplotlib.pyplot as plt
import datetime
import os
import math
import numpy as np
from dataclasses import dataclass, asdict

# ==========================================
# ハイパーパラメータの定義
# ==========================================
@dataclass
class EPropParams:
    n_inputs: int = 1
    n_lif: int = 100
    n_alif: int = 200
    n_outputs: int = 1
    
    # 時間定数関連
    dt: float = 1.0
    tau_mem: float = 20.0
    tau_r: float = 50.0
    tau_d: float = 125.0
    tau_alif: float = 2000.0
    t_ref: float = 5.0
    
    # ニューロンモデル関連
    v_th: float = 0.6
    gamma_d: float = 0.3
    beta_alif: float = 0.0174
    
    # ネットワーク構造・ノイズ関連
    g: float = 150.0
    density: float = 0.01
    tau_s: float = 250.0
    sigma_s: float = 1.0
    mu_s: float = 0.0
    I_bias: float = 0.02
    
    # 最適化関連 (Predictive E-prop)
    eta_rec: float = 0.0004
    eta_out: float = 0.0004
    lambda_reg: float = 2.0
    lambda_w: float = 5e-6
    t_delay: int = 10
    f_star: float = 0.01
    
    # タスク・シグナル関連 (Sine wave)
    A: float = 0.4
    T: float = 1000.0
    phi: float = 0.0
    c: float = 0.0
    
    # シミュレーションフェーズ長
    len_train: int = 10000
    len_error: int = 5000
    len_free: int = 5000
    epochs: int = 50
    
    @property
    def total_time(self) -> int:
        return self.len_train + self.len_error + self.len_free
    
    @property
    def n_neurons(self) -> int:
        return self.n_lif + self.n_alif

# ==========================================
# ニューロンモデル (GPU最適化のためLIF/ALIFを統合)
# ==========================================
class SpikingLayer:
    def __init__(self, p: EPropParams, device: torch.device):
        self.p = p
        self.device = device
        self.N = p.n_neurons
        
        # 減衰定数の事前計算
        self.alpha = math.exp(-p.dt / p.tau_mem)
        self.decay_r = math.exp(-p.dt / p.tau_r)
        self.decay_d = math.exp(-p.dt / p.tau_d)
        self.c_rd = 1.0 / (p.tau_r * p.tau_d)
        
        # ベクトル化された ALIF/LIF パラメータ
        self.beta = torch.zeros(self.N, device=device)
        self.beta[p.n_lif:] = p.beta_alif
        
        self.rho = torch.zeros(self.N, device=device)
        self.rho[p.n_lif:] = math.exp(-p.dt / p.tau_alif)
        
        self.reset_state()

    def reset_state(self):
        self.v = torch.zeros(self.N, device=self.device)
        self.a = torch.zeros(self.N, device=self.device)
        self.z = torch.zeros(self.N, device=self.device)
        self.z_bar = torch.zeros(self.N, device=self.device)
        self.z_bar_bar = torch.zeros(self.N, device=self.device)
        self.q = torch.full((self.N,), self.c_rd, device=self.device)
        self.psi = torch.zeros(self.N, device=self.device)
        self.b = torch.full((self.N,), self.p.v_th, device=self.device)
        self.ref_counts = torch.zeros(self.N, device=self.device)

    def step(self, input_current: torch.Tensor):
        # 適応閾値の更新 (LIF部分は rho=0, beta=0 なので影響なし)
        self.a = self.rho * self.a + self.z
        b_next = self.p.v_th + self.beta * self.a
        
        v_next_raw = self.alpha * self.v + input_current
        
        # サロゲート勾配 (Pseudo-derivative)
        self.psi = (self.p.gamma_d / self.p.v_th) * torch.relu(1.0 - torch.abs((v_next_raw - b_next) / self.p.v_th))
        
        is_refractory = self.ref_counts > 0
        self.z = ((v_next_raw > b_next) & (~is_refractory)).float()
        
        v_next = v_next_raw - self.z * b_next
        v_next = torch.where(is_refractory, self.v, v_next)
        
        self.ref_counts = torch.relu(self.ref_counts - self.p.dt)
        self.ref_counts = torch.where(self.z > 0, torch.tensor(self.p.t_ref, device=self.device), self.ref_counts)
        
        # 二重フィルタの更新（順序を最適化し、.clone()を削減）
        self.z_bar_bar = self.decay_d * self.z_bar_bar + self.q
        self.q = self.decay_r * self.q + self.z * self.c_rd
        self.z_bar = self.alpha * self.z_bar + self.z_bar_bar
        
        self.v = v_next
        self.b = b_next


# ==========================================
# 最適化手法 (Predictive E-prop) - テンソル並列化
# ==========================================
class EPropOptimizer:
    def __init__(self, p: EPropParams, layer: SpikingLayer, device: torch.device):
        self.p = p
        self.layer = layer
        self.device = device
        self.N = p.n_neurons
        
        self.beta_col = layer.beta.unsqueeze(1)
        self.rho_col = layer.rho.unsqueeze(1)
        
        self.reset_traces()

    def reset_traces(self):
        self.e_trace = torch.zeros((self.N, self.N), device=self.device)
        self.eps_b = torch.zeros((self.N, self.N), device=self.device)
        
        self.q_e = torch.full((self.N, self.N), self.layer.c_rd, device=self.device)
        self.e_bar_bar = torch.zeros((self.N, self.N), device=self.device)
        self.e_trace_sum_t_delays = torch.zeros((self.N, self.N), device=self.device)
        
        self.grad_w_rec = torch.zeros((self.N, self.N), device=self.device)
        self.grad_w_out = torch.zeros((self.p.n_outputs, self.N), device=self.device)
        self.grad_w_rec_accum = torch.zeros((self.N, self.N), device=self.device)
        self.grad_w_out_accum = torch.zeros((self.p.n_outputs, self.N), device=self.device)
        
        self.spike_buffer = torch.zeros(self.N, device=self.device)
        self.step_counter = 0

    def update_traces_and_gradients(self, z_bar_prev: torch.Tensor, z_bar_prev_prev: torch.Tensor, 
                                    psi_prev: torch.Tensor, L_t: torch.Tensor, error_t: torch.Tensor):
        
        psi_prev_col = psi_prev.unsqueeze(1)
        z_bb_row = z_bar_prev_prev.unsqueeze(0)
        psi_curr_col = self.layer.psi.unsqueeze(1)
        z_b_row = z_bar_prev.unsqueeze(0)
        
        # ALIF用のトレース更新 (LIFの行はbeta=0により自動的に消去される)
        self.eps_b = psi_prev_col * z_bb_row + (self.rho_col - self.beta_col * psi_prev_col) * self.eps_b
        
        # 適格性トレース
        self.e_trace = psi_curr_col * (z_b_row - self.beta_col * self.eps_b)
        
        # 二重フィルタ処理（順序を最適化し、.clone()を削減）
        self.e_bar_bar = self.layer.decay_d * self.e_bar_bar + self.q_e
        self.q_e = self.layer.decay_r * self.q_e + self.e_trace * self.layer.c_rd
        
        self.e_trace_sum_t_delays += self.e_trace
        
        # 勾配の蓄積 (W_rec と W_out)
        self.grad_w_rec_accum += L_t.unsqueeze(1) * self.e_bar_bar
        self.grad_w_out_accum += torch.outer(error_t, self.layer.z_bar)
        
        self.spike_buffer += self.layer.z
        self.step_counter += 1
        
        if self.step_counter >= self.p.t_delay:
            self.grad_w_rec = self.grad_w_rec_accum.clone()
            self.grad_w_out = self.grad_w_out_accum.clone()

            f_bar = self.spike_buffer / self.p.t_delay
            reg_signal = (self.p.lambda_reg / self.p.t_delay) * (f_bar - self.p.f_star)
            
            # ホメオスタシス正則化勾配の追加
            self.grad_w_rec += reg_signal.unsqueeze(1) * self.e_trace_sum_t_delays
            
            self.spike_buffer.fill_(0)
            self.e_trace_sum_t_delays.fill_(0)
            self.grad_w_rec_accum.fill_(0)
            self.grad_w_out_accum.fill_(0)
            self.step_counter = 0
            return True
            
        return False

    def apply_weight_update(self, w_rec: torch.Tensor, w_out: torch.Tensor):
        w_rec -= self.p.eta_rec * self.grad_w_rec + self.p.eta_rec * (2 * self.p.lambda_w) * w_rec
        w_out -= self.p.eta_out * self.grad_w_out + self.p.eta_out * (2 * self.p.lambda_w) * w_out
        
        self.grad_w_rec.fill_(0)
        self.grad_w_out.fill_(0)
        return w_rec, w_out


# ==========================================
# ネットワーク全体モデル
# ==========================================
class PredictiveEPropNet:
    def __init__(self, p: EPropParams, device: torch.device):
        self.p = p
        self.device = device
        self.N = p.n_neurons
        
        self.layer = SpikingLayer(p, device)
        
        self.w_in = torch.randn(self.N, p.n_inputs, device=device) / math.sqrt(p.g)
        self.w_fb = torch.randn(self.N, p.n_outputs, device=device) / math.sqrt(p.g)
        
        mask = (torch.rand(self.N, self.N, device=device) < p.density).float()
        mask.fill_diagonal_(0.0)
        self.w_rec_mask = mask
        
        self.w_rec = (torch.randn(self.N, self.N, device=device) / math.sqrt(self.N)) * self.w_rec_mask
        self.w_out = torch.randn(p.n_outputs, self.N, device=device) / math.sqrt(self.N)
        self.B = torch.randn(self.N, p.n_outputs, device=device) / math.sqrt(self.N)
        
        self.optimizer = EPropOptimizer(p, self.layer, device)
        
    def reset_state(self):
        self.layer.reset_state()
        self.optimizer.reset_traces()

    def generate_ou_noise_tensor(self, timesteps: int) -> torch.Tensor:
        """GPU上で全ステップのOUノイズを事前生成"""
        decay = math.exp(-self.p.dt / self.p.tau_s)
        noise = torch.randn(timesteps, self.N, device=self.device)
        s = torch.zeros(self.N, device=self.device)
        
        s_history = torch.empty(timesteps, self.N, device=self.device)
        for t in range(timesteps):
            s = self.p.mu_s + (s - self.p.mu_s) * decay + self.p.sigma_s * (self.p.dt / self.p.tau_s) * noise[t]
            s_history[t] = s
        return s_history

    def run_full_epoch(self, target_signal: torch.Tensor):
        """1エポック分のシミュレーションを実行し、出力と各フェーズの平均発火率を返す"""
        self.reset_state()
        timesteps = target_signal.shape[0]
        
        outputs = torch.zeros(timesteps, self.p.n_outputs, device=self.device)
            
        z_bar_prev = torch.zeros(self.N, device=self.device)
        z_bar_prev_prev = torch.zeros(self.N, device=self.device)
        psi_prev = torch.zeros(self.N, device=self.device)
        
        s_all = self.generate_ou_noise_tensor(timesteps)
        
        spikes_train = 0.0
        spikes_error = 0.0
        spikes_free  = 0.0
        
        # 学習ループは手動勾配計算のため autograd を無効化してメモリを節約
        with torch.no_grad():
            for t in range(timesteps):
                if t < self.p.len_train:
                    phase = "training"
                    xi = 1.0
                elif t < self.p.len_train + self.p.len_error:
                    phase = "error-driven"
                    xi = 1.0
                else:
                    phase = "free-running"
                    xi = 0.0

                x_t = target_signal[t]
                y_t = self.w_out @ self.layer.z_bar_bar
                outputs[t] = y_t
                
                d_t = y_t - x_t
                s_t = s_all[t]
                
                i_rec = self.w_rec @ self.layer.z_bar_bar
                i_in = self.w_in @ d_t
                i_fb = self.w_fb @ y_t
                
                total_current = xi * i_in + i_rec + i_fb + self.p.I_bias # + s_t
                
                self.layer.step(total_current)
                
                # スパイクの集計 (Hz計算用)
                total_spikes_t = self.layer.z.sum().item()
                if phase == "training":
                    spikes_train += total_spikes_t
                elif phase == "error-driven":
                    spikes_error += total_spikes_t
                else:
                    spikes_free += total_spikes_t
                
                if phase == "training":
                    L_t = self.B @ d_t
                    should_update = self.optimizer.update_traces_and_gradients(
                        z_bar_prev, z_bar_prev_prev, psi_prev, L_t, d_t
                    )
                    
                    if should_update:
                        self.w_rec, self.w_out = self.optimizer.apply_weight_update(self.w_rec, self.w_out)
                        self.w_rec *= self.w_rec_mask
                    
                    z_bar_prev_prev.copy_(z_bar_prev)
                    z_bar_prev.copy_(self.layer.z_bar)
                    psi_prev.copy_(self.layer.psi)
                    
                else:
                    # テストフェーズ中もホメオスタシスのためのバッファ更新を行う(重み更新はしない)
                    self.optimizer.spike_buffer += self.layer.z
                    self.optimizer.step_counter += 1
                    if self.optimizer.step_counter >= self.p.t_delay:
                        self.optimizer.spike_buffer.fill_(0)
                        self.optimizer.step_counter = 0

        # 全ニューロン・フェーズ長に対する平均発火率 (Hz) の計算
        hz_train = (spikes_train / (self.p.len_train * (self.p.dt / 1000.0))) / self.N
        hz_error = (spikes_error / (self.p.len_error * (self.p.dt / 1000.0))) / self.N
        hz_free  = (spikes_free / (self.p.len_free * (self.p.dt / 1000.0))) / self.N
                        
        return outputs, hz_train, hz_error, hz_free


# ==========================================
# タスク実行とグラフ描画
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ハイパーパラメータの設定
    params = EPropParams()
    
    # ターゲット波形の生成
    t_all = torch.arange(0, params.total_time, params.dt, device=device)
    target_all = (params.A * torch.sin((2 * math.pi / params.T) * t_all + params.phi) + params.c).unsqueeze(1)
    
    net = PredictiveEPropNet(params, device=device)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output", exist_ok=True)
    
    out_all_cpu = None
    target_all_cpu = target_all.cpu().numpy()
    t_all_cpu = t_all.cpu().numpy()
    
    # 発火率記録用リスト
    hz_train_list = []
    hz_error_list = []
    hz_free_list  = []
    
    for epoch in range(params.epochs):
        print(f"--- Epoch {epoch+1}/{params.epochs} ---")
        
        out_all, hz_train, hz_error, hz_free = net.run_full_epoch(target_all)
        
        out_all_cpu = out_all.cpu().numpy()
        
        # 損失の計算
        loss_train = torch.mean((out_all[:params.len_train] - target_all[:params.len_train])**2).item()
        loss_error = torch.mean((out_all[params.len_train:params.len_train+params.len_error] - target_all[params.len_train:params.len_train+params.len_error])**2).item()
        loss_free = torch.mean((out_all[params.len_train+params.len_error:] - target_all[params.len_train+params.len_error:])**2).item()
        
        hz_train_list.append(hz_train)
        hz_error_list.append(hz_error)
        hz_free_list.append(hz_free)
        
        print(f"  Training Loss: {loss_train:.5f} | Firing Rate: {hz_train:.2f} Hz")
        print(f"  Error-driven Loss: {loss_error:.5f} | Firing Rate: {hz_error:.2f} Hz")
        print(f"  Free-running Loss: {loss_free:.5f} | Firing Rate: {hz_free:.2f} Hz")
    
    # ==========================================
    # 最終結果のプロット: 時系列生成結果
    # ==========================================
    img_filename = f"outputFire/predictive_eprop_result_{timestamp}.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    axes[0].plot(t_all_cpu[:params.len_train], target_all_cpu[:params.len_train], label="Target", color="black", linestyle="--")
    axes[0].plot(t_all_cpu[:params.len_train], out_all_cpu[:params.len_train], label="Output", color="blue", alpha=0.7)
    axes[0].set_title("Training Phase")
    axes[0].set_ylabel("Signal")
    axes[0].legend(loc="upper right")
    
    start_err, end_err = params.len_train, params.len_train + params.len_error
    axes[1].plot(t_all_cpu[start_err:end_err], target_all_cpu[start_err:end_err], label="Target", color="black", linestyle="--")
    axes[1].plot(t_all_cpu[start_err:end_err], out_all_cpu[start_err:end_err], label="Output", color="orange", alpha=0.7)
    axes[1].set_title("Error-driven Phase")
    axes[1].set_ylabel("Signal")
    axes[1].legend(loc="upper right")
    
    axes[2].plot(t_all_cpu[end_err:], target_all_cpu[end_err:], label="Target", color="black", linestyle="--")
    axes[2].plot(t_all_cpu[end_err:], out_all_cpu[end_err:], label="Output", color="green", alpha=0.7)
    axes[2].set_title("Free-running Phase")
    axes[2].set_xlabel("Time [ms]")
    axes[2].set_ylabel("Signal")
    axes[2].legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(img_filename)
    print(f"\n推論結果のプロットを画像ファイル '{img_filename}' として保存しました。")
    
    # ==========================================
    # 最終結果のプロット: ネットワーク平均発火率の推移 (Fig 2C 相当)
    # ==========================================
    fr_filename = f"outputFire/predictive_eprop_firing_rate_{timestamp}.png"
    fig_fr, ax_fr = plt.subplots(figsize=(8, 5))
    
    epochs_arr = np.arange(1, params.epochs + 1)
    ax_fr.plot(epochs_arr, hz_train_list, label="Training", color="blue")
    ax_fr.plot(epochs_arr, hz_error_list, label="Error-driven", color="orange")
    ax_fr.plot(epochs_arr, hz_free_list, label="Free-running", color="green")
    
    ax_fr.set_xlabel("Epoch")
    ax_fr.set_ylabel("Firing Rate [Hz]")
    ax_fr.set_title("Average Firing Rate across Epochs")
    ax_fr.legend()
    ax_fr.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(fr_filename)
    print(f"発火率推移のプロットを画像ファイル '{fr_filename}' として保存しました。")

    # ハイパーパラメータのテキスト保存
    param_filename = f"outputFire/predictive_eprop_params_{timestamp}.txt"
    with open(param_filename, "w", encoding="utf-8") as f:
        f.write("スクリプト内で使用されたハイパーパラメータ一覧\n")
        f.write("=" * 60 + "\n")
        for key, value in asdict(params).items():
            f.write(f"{key}: {value}\n")
            
    print(f"スクリプトで使用されたパラメータ数値をテキストファイル '{param_filename}' として保存しました。")