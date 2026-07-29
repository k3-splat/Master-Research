import torch
import torch.nn as nn
import numpy as np

# 1. サロゲート勾配（Tent関数）の定義 (論文 Eq. 7)
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, b, v_th, gamma_d):
        ctx.save_for_backward(v, b)
        ctx.v_th = v_th
        ctx.gamma_d = gamma_d
        # スパイクの発生: 膜電位 v が閾値 b を超えたら 1、それ以外は 0
        out = (v >= b).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        v, b = ctx.saved_tensors
        v_th = ctx.v_th
        gamma_d = ctx.gamma_d
        # Tent関数による勾配近似 (Eq. 7)
        grad = (1.0 / v_th) * gamma_d * torch.clamp(1.0 - torch.abs((v - b) / v_th), min=0.0)
        return grad * grad_output, None, None, None

spike_function = SurrogateSpike.apply

# 2. Predictive E-prop ネットワークの基本モジュール
class PredictiveEPropRSNN(nn.Module):
    def __init__(self, input_dim=620, n_lif=100, n_alif=200, dt=1.0):
        super(PredictiveEPropRSNN, self).__init__()
        self.dt = dt
        self.input_dim = input_dim
        self.N = n_lif + n_alif  # 全ニューロン数 (300)
        self.n_lif = n_lif
        self.n_alif = n_alif
        
        # --- 論文 Table SI / SII に基づくハイパーパラメータ ---
        self.tau_mem = 20.0       # 膜電位の時定数 (ms)
        self.tau_alif = 2000.0    # ALIFの適応閾値の時定数 (ms)
        self.tau_r = 50.0         # ダブル指数関数の立ち上がり時定数
        self.tau_d = 125.0        # ダブル指数関数の減衰時定数
        self.v_th = 0.6           # 基本発火閾値 (mV)
        self.I_bias = 0.02        # バイアス電流
        self.gamma_d = 0.3        # サロゲート勾配の高さ
        self.g = 35.0             # 入力/フィードバック重みのゲイン (Lorenzタスク設定を参考)
        self.beta = torch.cat([torch.zeros(n_lif), torch.ones(n_alif) * 1.7]) # ALIF用係数 (ヒューリスティック定数)
        
        # 減衰定数の計算
        self.alpha = np.exp(-self.dt / self.tau_mem)
        self.rho = np.exp(-self.dt / self.tau_alif)
        self.kappa_r = np.exp(-self.dt / self.tau_r)
        self.kappa_d = np.exp(-self.dt / self.tau_d)
        
        # --- 重み行列の初期化 (Sec 2.1) ---
        # 固定重み (W_in, W_fb) : N(0, 1) / sqrt(g)
        self.W_in = nn.Parameter(torch.randn(self.N, self.input_dim) / np.sqrt(self.g), requires_grad=False)
        self.W_fb = nn.Parameter(torch.randn(self.N, self.input_dim) / np.sqrt(self.g), requires_grad=False)
        
        # 学習重み (W_rec, W_out) : N(0, 1) / sqrt(N_rec)
        self.W_rec = nn.Parameter(torch.randn(self.N, self.N) / np.sqrt(self.N), requires_grad=True)
        self.W_out = nn.Parameter(torch.randn(self.input_dim, self.N) / np.sqrt(self.N), requires_grad=True)
        
        # Broadcast Alignment用ランダム行列 (B_ij) (固定)
        self.B_align = nn.Parameter(torch.randn(self.N, self.input_dim), requires_grad=False)

    def forward(self, x_seq, free_running=False):
        """
        x_seq: ターゲットのスパイク列。形状は (seq_len, batch_size, input_dim) を想定。
               今回はバッチサイズ1で、(119999, 1, 620) のような入力を想定します。
        free_running: Trueの場合、外部入力（予測誤差）を遮断し、自律生成モードになります。
        """
        seq_len = x_seq.size(0)
        batch_size = x_seq.size(1)
        
        # 1. 状態変数の初期化 (batch_size, N=300)
        v = torch.zeros(batch_size, self.N, device=x_seq.device)     # 膜電位
        a = torch.zeros(batch_size, self.N, device=x_seq.device)     # ALIFの適応閾値用状態
        z = torch.zeros(batch_size, self.N, device=x_seq.device)     # スパイクの有無 (0 or 1)
        z_bar = torch.zeros(batch_size, self.N, device=x_seq.device) # ダブル指数フィルタ適用後のスパイク
        q = torch.zeros(batch_size, self.N, device=x_seq.device)     # フィルタの補助変数
        
        # ログ保存用リスト
        y_seq = [] # ネットワークの予測出力
        z_seq = [] # リカレント層のスパイク
        d_seq = [] # 予測誤差
        
        # ξ (xi): エラー入力の有無を制御 (Eq.1)。Free-running時は0になる。
        xi = 0.0 if free_running else 1.0
        
        # --- 時間ステップごとのループ処理 ---
        for t in range(seq_len):
            xt = x_seq[t] # 現在のターゲット信号 (1, 620)
            
            # 2. ネットワークの予測出力 y^t と 予測誤差 d^t の計算
            yt = torch.matmul(z_bar, self.W_out.t())
            dt = yt - xt
            
            # 3. 閾値 b^t の計算 (Eq.1の直後)
            # LIFニューロンはbeta=0なので固定、ALIFニューロンはbeta>0なので適応的に変化します
            b = self.v_th + self.beta * a
            
            # 4. 膜電位 v^{t+1} の計算 (Eq.1)
            # ※ノイズ項(s)は、論文ではOrnstein-Uhlenbeck過程ですが、まずはシンプルなガウスノイズで実装します
            noise = torch.randn_like(v) * 0.01 
            
            # 各結合からの入力電流を計算
            i_in = xi * torch.matmul(dt, self.W_in.t()) # 予測誤差からの入力
            i_rec = torch.matmul(z_bar, self.W_rec.t()) # リカレント結合からの入力
            i_fb = torch.matmul(yt, self.W_fb.t())      # 自身の予測からのフィードバック
            
            # リセット機構 (z * b) を含めた膜電位の更新
            v = self.alpha * v + i_in + i_rec + i_fb + self.I_bias + noise - z * b
            
            # 5. スパイク z^{t+1} の生成 (フェーズ1で定義したSurrogate Gradientを使用)
            z = spike_function(v, b, self.v_th, self.gamma_d)
            
            # 6. 適応閾値状態 a^{t+1} の更新 (Eq.1の直後)
            a = self.rho * a + z
            
            # 7. ダブル指数フィルタの更新 (Eq.2)
            z_bar = self.kappa_d * z_bar + q
            q = self.kappa_r * q + (1.0 / (self.tau_r * self.tau_d)) * z
            
            # 記録
            y_seq.append(yt)
            z_seq.append(z)
            d_seq.append(dt)
            
        # リストをテンソルに変換して返す
        return torch.stack(y_seq), torch.stack(z_seq), torch.stack(d_seq)