import torch
import torch.nn as nn
import numpy as np

# ========================================================
# 1. サロゲート勾配（Tent関数）の定義
# ========================================================
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, b, v_th, gamma_d):
        ctx.save_for_backward(v, b)
        ctx.v_th = v_th
        ctx.gamma_d = gamma_d
        out = (v >= b).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        v, b = ctx.saved_tensors
        v_th = ctx.v_th
        gamma_d = ctx.gamma_d
        grad = (1.0 / v_th) * gamma_d * torch.clamp(1.0 - torch.abs((v - b) / v_th), min=0.0)
        return grad * grad_output, None, None, None

spike_function = SurrogateSpike.apply

# ========================================================
# 2. Predictive E-prop ネットワーク (Version 2)
# ========================================================
class PredictiveEPropRSNN(nn.Module):
    def __init__(self, input_dim=620, n_lif=100, n_alif=200, dt=1.0):
        super(PredictiveEPropRSNN, self).__init__()
        self.dt = dt
        self.input_dim = input_dim
        self.N = n_lif + n_alif
        self.n_lif = n_lif
        self.n_alif = n_alif

        # --- 基本パラメータ (Table SIに準拠) ---
        self.tau_mem = 20.0       
        self.tau_alif = 2000.0    
        self.tau_r = 50.0         
        self.tau_d = 125.0        
        self.v_th = 0.6           
        self.I_bias = 0.02        
        self.gamma_d = 0.3        
        self.g = 35.0             
        self.beta = torch.cat([torch.zeros(n_lif), torch.ones(n_alif) * 1.7]) 

        # --- 恒常性正則化パラメータ ---
        self.lambda_reg = 2.0      # 論文 Table SI に基づく強度
        self.f_target = 0.01       # ターゲット発火率 (10Hz = 0.01 spikes/ms)
        self.t_delay = 10          # 更新間隔 (10ms)

        # --- O-Uノイズ パラメータ (★バグ修正: sigma_s を 論文通りの 1.0 に設定) ---
        self.tau_s = 250.0         
        self.sigma_s = 1.0         # 論文 Table SI に準拠
        self.alpha_s = np.exp(-self.dt / self.tau_s)
        self.sigma_step = self.sigma_s * np.sqrt(1.0 - self.alpha_s**2)

        # 減衰定数
        self.alpha = np.exp(-self.dt / self.tau_mem)
        self.rho = np.exp(-self.dt / self.tau_alif)
        self.kappa_r = np.exp(-self.dt / self.tau_r)
        self.kappa_d = np.exp(-self.dt / self.tau_d)
        
        # --- Sparsity 0.99 の実装 (1%の結合のみ保持) ---
        self.sparsity = 0.99
        p_conn = 1.0 - self.sparsity # 接続確率は0.01
        
        # 固定重み
        self.W_in = nn.Parameter(torch.randn(self.N, self.input_dim) / np.sqrt(self.g), requires_grad=False)
        self.W_fb = nn.Parameter(torch.randn(self.N, self.input_dim) / np.sqrt(self.g), requires_grad=False)
        self.B_align = nn.Parameter(torch.randn(self.N, self.input_dim), requires_grad=False)
        
        # 学習重み (W_rec / W_out)
        self.W_rec = nn.Parameter(torch.randn(self.N, self.N) / np.sqrt(self.N), requires_grad=True)
        self.W_out = nn.Parameter(torch.randn(self.input_dim, self.N) / np.sqrt(self.N), requires_grad=True)
        
        # Sparsityマスク
        mask = (torch.rand(self.N, self.N) < p_conn).float()
        self.register_buffer('W_rec_mask', mask)

    def forward(self, x_seq, free_running=False, state=None, update_grads=True):
        """
        順伝播処理
        Args:
            x_seq: 目標信号テンソル (seq_len, batch_size, input_dim)
            free_running: Trueの場合、外部予測誤差入力を遮断 (xi = 0)
            state: ニューロン状態辞書。None の場合はゼロ初期化し、前回の状態がある場合は引き継ぎます
            update_grads: Trueの場合、W_rec と W_out の e-prop 手動勾配を計算して .grad に代入
        """
        seq_len = x_seq.size(0)
        batch_size = x_seq.size(1)
        device = x_seq.device
        
        # 状態の初期化または復元 (3フェーズ間での連続的な状態引き継ぎ用)
        if state is None:
            v = torch.zeros(batch_size, self.N, device=device)
            a = torch.zeros(batch_size, self.N, device=device)
            z = torch.zeros(batch_size, self.N, device=device)
            z_bar = torch.zeros(batch_size, self.N, device=device)
            q = torch.zeros(batch_size, self.N, device=device)
            ou_noise = torch.zeros(batch_size, self.N, device=device)
            trace_a = torch.zeros(batch_size, self.N, self.N, device=device)
        else:
            v = state['v'].clone()
            a = state['a'].clone()
            z = state['z'].clone()
            z_bar = state['z_bar'].clone()
            q = state['q'].clone()
            ou_noise = state['ou_noise'].clone()
            trace_a = state['trace_a'].clone()
            
        y_seq, z_seq, d_seq = [], [], []
        xi = 0.0 if free_running else 1.0
        
        grad_W_rec = torch.zeros_like(self.W_rec)
        grad_W_out = torch.zeros_like(self.W_out)
        beta_exp = self.beta.unsqueeze(0).unsqueeze(2).to(device)

        z_buffer = []
        e_trace_buffer = []
        
        # スパース接続を適用した recurrent 重み
        W_rec_sparse = self.W_rec * self.W_rec_mask

        for t in range(seq_len):
            xt = x_seq[t]
            
            yt = torch.matmul(z_bar, self.W_out.t())
            dt = yt - xt
            b = self.v_th + self.beta.to(device) * a
            
            v_scaled = (v - b) / self.v_th
            psi = (self.gamma_d / self.v_th) * torch.clamp(1.0 - torch.abs(v_scaled), min=0.0)
            
            # Ornstein-Uhlenbeck ノイズ更新
            ou_noise = self.alpha_s * ou_noise + self.sigma_step * torch.randn_like(v)
            
            i_in = xi * torch.matmul(dt, self.W_in.t())
            i_rec = torch.matmul(z_bar, W_rec_sparse.t()) 
            i_fb = torch.matmul(yt, self.W_fb.t())
            
            v = self.alpha * v + i_in + i_rec + i_fb + self.I_bias + ou_noise - z * b
            
            z = spike_function(v, b, self.v_th, self.gamma_d)
            a = self.rho * a + z
            
            z_prev_bar = z_bar.clone() 
            z_bar = self.kappa_d * z_bar + q
            q = self.kappa_r * q + (1.0 / (self.tau_r * self.tau_d)) * z
            
            if not free_running and update_grads:
                # e-propによる局所的な勾配情報の蓄積 (Eq. 3, 5, 9, 10)
                L_t = torch.matmul(dt, self.B_align.t()) / (self.tau_r * self.tau_d)
                
                psi_exp = psi.unsqueeze(2)           
                z_pre_exp = z_prev_bar.unsqueeze(1)  
                
                trace_a = (self.rho - beta_exp * psi_exp) * trace_a + psi_exp * z_pre_exp
                e_trace = psi_exp * (z_pre_exp - beta_exp * trace_a)
                
                L_t_exp = L_t.unsqueeze(2)
                grad_W_rec += torch.sum(L_t_exp * e_trace, dim=0) 
                grad_W_out += torch.matmul(dt.t(), z_bar)

                z_buffer.append(z)
                e_trace_buffer.append(e_trace)

                if len(z_buffer) == self.t_delay:
                    f_bar = torch.mean(torch.stack(z_buffer), dim=0) 
                    
                    # ★バグ修正: 論文 Eq. S3 の微分に基づき、lambda_reg を掛け算して適用
                    f_err = self.lambda_reg * (f_bar - self.f_target)
                    L_reg = f_err / self.t_delay

                    sum_e_trace = torch.sum(torch.stack(e_trace_buffer), dim=0)
                    grad_W_rec += torch.sum(L_reg.unsqueeze(2) * sum_e_trace, dim=0)

                    z_buffer = []
                    e_trace_buffer = []

            y_seq.append(yt)
            z_seq.append(z)
            d_seq.append(dt)

        # 残分バッファの正則化
        if not free_running and update_grads and len(z_buffer) > 0:
            f_bar = torch.mean(torch.stack(z_buffer), dim=0)
            f_err = self.lambda_reg * (f_bar - self.f_target)
            L_reg = f_err / len(z_buffer)
            sum_e_trace = torch.sum(torch.stack(e_trace_buffer), dim=0)
            grad_W_rec += torch.sum(L_reg.unsqueeze(2) * sum_e_trace, dim=0)

        # 勾配を parameters.grad に手動割り当て
        if not free_running and update_grads:
            # マスクによるスパース性の維持 (勾配も1%の結合のみに限定)
            self.W_rec.grad = grad_W_rec * self.W_rec_mask
            self.W_out.grad = grad_W_out
            
        # 最終状態をまとめて返却し、次のフェーズに持ち越せるようにする
        next_state = {
            'v': v,
            'a': a,
            'z': z,
            'z_bar': z_bar,
            'q': q,
            'ou_noise': ou_noise,
            'trace_a': trace_a
        }
            
        return torch.stack(y_seq), torch.stack(z_seq), torch.stack(d_seq), next_state
