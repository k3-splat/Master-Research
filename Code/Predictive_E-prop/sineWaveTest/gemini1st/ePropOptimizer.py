import torch
import torch.nn as nn
import datetime

class EPropOptimizer:
    """
    Predictive E-propアルゴリズムによる学習および正則化を管理するオプティマイザー
    """
    def __init__(self, 
                 network, 
                 lr=0.0004, 
                 gamma_d=0.3, 
                 lambda_reg=2.0, 
                 lambda_w=5e-6, 
                 f_target=0.01,  # 10 Hz = 0.01 spikes/ms
                 t_delay=10):
        
        self.network = network
        self.lr = lr
        self.gamma_d = gamma_d
        self.lambda_reg = lambda_reg
        self.lambda_w = lambda_w
        self.f_target = f_target
        self.t_delay = t_delay
        
        self.n_total = network.n_total
        
        # Broadcast alignment用のランダム固定行列 B_ij を初期化
        self.B = torch.randn(self.n_total, 1)
        
        self.reset_states()

    def reset_states(self):
        """学習に必要な適格度トレースおよび蓄積変数をリセットする"""
        # トレース用変数
        self.eps_b = torch.zeros(self.n_total, self.n_total)
        self.e_trace = torch.zeros(self.n_total, self.n_total)
        
        # 過去の状態を保存する変数（1ステップ前、2ステップ前）
        self.psi_prev = torch.zeros(self.n_total)
        self.z_bar_prev1 = torch.zeros(self.n_total)
        self.z_bar_prev2 = torch.zeros(self.n_total)
        
        # 重み更新の蓄積用テンソル
        self.dW_rec = torch.zeros(self.n_total, self.n_total)
        self.dW_out = torch.zeros(1, self.n_total)
        
        # 正則化（発火率）用のスパイク蓄積
        self.spike_count = torch.zeros(self.n_total)
        self.step_counter = 0

    def step(self, x_t, y_t, v_t, b_t, z_t, z_bar_t, is_training=True):
        """
        1タイムステップ分の学習処理（トレース更新および勾配蓄積）
        
        Args:
            x_t (torch.Tensor): ターゲット信号
            y_t (torch.Tensor): ネットワークの予測出力
            v_t (torch.Tensor): 膜電位
            b_t (torch.Tensor): 閾値
            z_t (torch.Tensor): 現在のスパイク
            z_bar_t (torch.Tensor): フィルター処理済みスパイク
            is_training (bool): 重みを更新するかどうか（Error-drivenフェーズ用）
        """
        # 1. 予測誤差の計算
        d_t = y_t - x_t
        
        # 2. サロゲート勾配（テント関数）psi_i^t の計算
        # psi_i^t = (gamma_d / v_th) * max(0, 1 - |(v_i^t - b_i^t) / v_th|)
        v_th = self.network.v_th
        abs_diff = torch.abs((v_t - b_t) / v_th)
        psi_t = (self.gamma_d / v_th) * torch.clamp(1.0 - abs_diff, min=0.0)
        
        if is_training:
            # 3. ALIFニューロン用の適応閾値トレース eps_b の更新
            # eps_{ij,b}^t = psi_i^{t-1} * z_bar_j^{t-2} + (rho - beta * psi_i^{t-1}) * eps_{ij,b}^{t-1}
            term1 = torch.outer(self.psi_prev, self.z_bar_prev2)
            beta_matrix = self.network.beta.unsqueeze(1) # [n_total, 1]
            term2_coef = self.network.rho - (beta_matrix * self.psi_prev.unsqueeze(1))
            self.eps_b = term1 + term2_coef * self.eps_b
            
            # 4. 全体の適格度トレース e_{ij}^t の計算
            # e_{ij}^t = psi_i^t * (z_bar_j^{t-1} - beta * eps_{ij,b}^t)
            z_bar_prev1_matrix = self.z_bar_prev1.unsqueeze(0).repeat(self.n_total, 1) # [n_total, n_total]
            e_trace = psi_t.unsqueeze(1) * (z_bar_prev1_matrix - beta_matrix * self.eps_b)
            
            # 5. 学習シグナル L_i^t の計算
            # L_i^t = B_ij * d_t * (1 / (tau_r * tau_d))
            L_t = self.B * d_t * self.network.filter_c
            
            # 6. 勾配の蓄積
            # dW_rec += L_i^t * e_{ij}^t * lr
            self.dW_rec += self.lr * L_t * e_trace
            
            # dW_out += d_t * z_bar_j^t * lr
            self.dW_out += self.lr * d_t * z_bar_t.unsqueeze(0)
            
            # 7. 正則化の適用 (t_delay ごと)
            self.spike_count += z_t
            self.step_counter += 1
            
            if self.step_counter >= self.t_delay:
                self.apply_updates_and_regularization()
                self.spike_count.zero_()
                self.step_counter = 0

        # 状態変数を次ステップへ持ち越し
        self.psi_prev = psi_t.clone()
        self.z_bar_prev2 = self.z_bar_prev1.clone()
        self.z_bar_prev1 = z_bar_t.clone()

    def apply_updates_and_regularization(self):
        """蓄積された勾配と正則化ペナルティを用いて重みを更新する"""
        
        # -- 予測誤差に関する更新 --
        with torch.no_grad():
            self.network.W_rec += self.dW_rec
            self.network.W_out += self.dW_out
            
            # -- 重み減衰 (Weight Decay) --
            # - lambda_w * W
            self.network.W_rec -= self.lambda_w * self.network.W_rec
            self.network.W_out -= self.lambda_w * self.network.W_out
            
            # -- 発火率の恒常性 (Homeostatic Plasticity) --
            # f_bar = spike_count / t_delay
            f_bar = self.spike_count / self.t_delay
            # 発火率エラー: (f_bar - f^*)
            f_error = f_bar - self.f_target
            
            # W_rec に対する正則化の勾配ペナルティを適用 (簡易的実装)
            # 過剰発火しているニューロンの入力重みを下げる
            reg_penalty = self.lambda_reg * f_error.unsqueeze(1).repeat(1, self.n_total)
            self.network.W_rec -= self.lr * reg_penalty
            
            # 更新用蓄積テンソルのリセット
            self.dW_rec.zero_()
            self.dW_out.zero_()


# ==========================================
# テスト実行コード (単体テスト用)
# ==========================================
if __name__ == "__main__":
    # ネットワーク層のモックを定義（SpikingNeuronLayerが同階層にある想定の簡略化）
    class MockNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_total = 300
            self.v_th = 0.6
            self.filter_c = 1.0 / (50.0 * 125.0)
            self.rho = torch.exp(torch.tensor(-1.0 / 2000.0))
            self.beta = torch.cat([torch.zeros(100), torch.ones(200)])
            
            self.W_rec = nn.Parameter(torch.randn(self.n_total, self.n_total) * 0.01, requires_grad=False)
            self.W_out = nn.Parameter(torch.randn(1, self.n_total) * 0.01, requires_grad=False)

    network = MockNetwork()
    optimizer = EPropOptimizer(network)
    
    time_steps = 20
    print(f"Initial W_out sum: {network.W_out.sum().item():.6f}")
    
    # 20ステップのダミーループを回してエラーが出ないかテスト
    for t in range(time_steps):
        # ダミーの状態ベクトルを生成
        x_t = torch.tensor([0.5])
        y_t = torch.tensor([0.1])
        v_t = torch.randn(network.n_total) * 0.2 + 0.5
        b_t = network.v_th + network.beta * 0.1
        z_t = (torch.rand(network.n_total) > 0.9).float()
        z_bar_t = torch.rand(network.n_total) * 0.05
        
        optimizer.step(x_t, y_t, v_t, b_t, z_t, z_bar_t, is_training=True)
        
    print(f"Updated W_out sum after 20 steps (2 regularizations): {network.W_out.sum().item():.6f}")
    print("EPropOptimizer module passed the sanity check without errors.")