import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os
import sys

# 同一ディレクトリの Network_v2 からのインポートを保証
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Network_v2 import PredictiveEPropRSNN

# シード値の固定による再現性の確保
torch.manual_seed(42)
np.random.seed(42)

# GPU/CPUデバイスの自動選択
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================================================
# 1. 論文に基づくサイン波データの生成 (全 20,000 ms)
# ========================================================
# 1 epoch の構成 (Table SI):
# - Training phase: 10,000 ms (0 ~ 10s)
# - Error-driven phase: 5,000 ms (10 ~ 15s)
# - Free-running phase: 5,000 ms (15 ~ 20s)
A = 0.4
T_period = 1000.0  # 周期 1000 ms
dt = 1.0

total_len = 20000  # 20秒
t_total = np.arange(0, total_len, dt)
target_signal = A * np.sin(2 * np.pi * t_total / T_period)

# PyTorch テンソル形状 (seq_len, batch_size, input_dim) -> (20000, 1, 1)
target_tensor = torch.tensor(target_signal, dtype=torch.float32).view(total_len, 1, 1).to(device)

# ========================================================
# 2. モデルの初期化とサイン波用パラメータの設定 (Table SI)
# ========================================================
model = PredictiveEPropRSNN(input_dim=1, n_lif=100, n_alif=200, dt=1.0).to(device)

# サイン波タスク固有のゲイン設定 (g = 150)
model.g = 150.0
model.W_in = nn.Parameter(torch.randn(model.N, model.input_dim, device=device) / np.sqrt(model.g), requires_grad=False)
model.W_fb = nn.Parameter(torch.randn(model.N, model.input_dim, device=device) / np.sqrt(model.g), requires_grad=False)

# オプティマイザ (学習対象： W_rec, W_out)
learning_rate = 0.0004
weight_decay = 5e-6
optimizer = torch.optim.Adam([model.W_rec, model.W_out], lr=learning_rate, weight_decay=weight_decay)

# ========================================================
# 3. 評価指標の定義 (ダウンサンプリングDTW距離)
# ========================================================
def compute_dtw(x, y, downsample_rate=10):
    """
    1D numpy 配列間の DTW 距離を計算します。
    ステップ数が長いため、計算負荷を抑える目的で適度にダウンサンプリングして動的計画法を適用します。
    """
    x_ds = x[::downsample_rate]
    y_ds = y[::downsample_rate]
    N = len(x_ds)
    M = len(y_ds)
    dp = np.zeros((N, M))
    dp[0, 0] = (x_ds[0] - y_ds[0])**2
    for i in range(1, N):
        dp[i, 0] = dp[i-1, 0] + (x_ds[i] - y_ds[0])**2
    for j in range(1, M):
        dp[0, j] = dp[0, j-1] + (x_ds[0] - y_ds[j])**2
        
    for i in range(1, N):
        for j in range(1, M):
            cost = (x_ds[i] - y_ds[j])**2
            dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
            
    return np.sqrt(dp[-1, -1]) / N

# ========================================================
# 4. 実験ループ (3フェーズ 50 epoch の順次シミュレーション)
# ========================================================
# ※注意：CPU環境などでテスト実行する際は、num_epochs = 3 〜 5 程度に減らすと短時間で終了します。
num_epochs = 50  # 論文基準は 50 epoch

history = {
    'epoch': [],
    'train_loss': [],
    'err_loss': [],
    'free_loss': [],
    'err_dtw': [],
    'free_dtw': [],
    'train_fr': [],
    'err_fr': [],
    'free_fr': []
}

print(f"Starting 3-Phase Simulation loop ({num_epochs} Epochs)...")

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    # 各 epoch の開始時にニューロン内部状態をゼロに初期化
    state = None
    
    # ---------------------------------------------
    # Phase 1: Training Phase (0 ~ 10,000 ms)
    # ---------------------------------------------
    optimizer.zero_grad()
    
    # ※メモリの急増(OOM)を防止するため、手動 e-prop 計算でも必ず no_grad コンテキスト内で順伝播させます
    with torch.no_grad():
        y_train, z_train, d_train, state = model(
            target_tensor[0:10000], 
            free_running=False, 
            state=state, 
            update_grads=True
        )
    
    # 蓄積された勾配を基に重みを更新
    optimizer.step()
    
    train_loss = 0.5 * torch.sum(d_train ** 2) / 10000
    train_fr = torch.mean(z_train).item() * 1000.0  # Hz
    
    # ---------------------------------------------
    # Phase 2: Error-driven Phase (10,000 ~ 15,000 ms)
    # ---------------------------------------------
    # 重みを固定 (update_grads=False)、エラー入力は供給 (free_running=False)
    with torch.no_grad():
        y_err, z_err, d_err, state = model(
            target_tensor[10000:15000], 
            free_running=False, 
            state=state, 
            update_grads=False
        )
    err_loss = 0.5 * torch.sum(d_err ** 2) / 5000
    err_fr = torch.mean(z_err).item() * 1000.0  # Hz
    
    # DTWの計算
    y_err_np = y_err.squeeze().cpu().numpy()
    target_err_np = target_signal[10000:15000]
    err_dtw = compute_dtw(y_err_np, target_err_np)
    
    # ---------------------------------------------
    # Phase 3: Free-running Phase (15,000 ~ 20,000 ms)
    # ---------------------------------------------
    # 重みを固定 (update_grads=False)、エラー入力を遮断 (free_running=True)
    with torch.no_grad():
        y_free, z_free, d_free, state = model(
            target_tensor[15000:20000], 
            free_running=True, 
            state=state, 
            update_grads=False
        )
    free_loss = 0.5 * torch.sum(d_free ** 2) / 5000
    free_fr = torch.mean(z_free).item() * 1000.0  # Hz
    
    # DTWの計算
    y_free_np = y_free.squeeze().cpu().numpy()
    target_free_np = target_signal[15000:20000]
    free_dtw = compute_dtw(y_free_np, target_free_np)
    
    # 履歴への記録
    history['epoch'].append(epoch + 1)
    history['train_loss'].append(train_loss.item())
    history['err_loss'].append(err_loss.item())
    history['free_loss'].append(free_loss.item())
    history['err_dtw'].append(err_dtw)
    history['free_dtw'].append(free_dtw)
    history['train_fr'].append(train_fr)
    history['err_fr'].append(err_fr)
    history['free_fr'].append(free_fr)
    
    epoch_time = time.time() - epoch_start
    print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
          f"Train Loss (MSE): {train_loss.item():.6f} (FR: {train_fr:.1f} Hz) | "
          f"Err DTW: {err_dtw:.6f} (FR: {err_fr:.1f} Hz) | "
          f"Free DTW: {free_dtw:.6f} (FR: {free_fr:.1f} Hz) | "
          f"Time: {epoch_time:.1f} s")

print("Training Completed Successfully!")

# ========================================================
# 5. 学習結果の可視化と図の保存
# ========================================================
print("Generating evaluation plots...")

# 最終訓練済みモデルでの20秒シミュレーション
final_state = None
with torch.no_grad():
    y_t, z_t, _, final_state = model(target_tensor[0:10000], free_running=False, state=final_state, update_grads=False)
    y_e, z_e, _, final_state = model(target_tensor[10000:15000], free_running=False, state=final_state, update_grads=False)
    y_f, z_f, _, final_state = model(target_tensor[15000:20000], free_running=True, state=final_state, update_grads=False)

y_all = torch.cat([y_t, y_e, y_f], dim=0).squeeze().cpu().numpy()

plt.figure(figsize=(12, 10))

# ① 出力波形
plt.subplot(3, 1, 1)
plt.plot(t_total, target_signal, label="Target (Sine)", color="blue", linestyle="--")
plt.plot(t_total, y_all, label="Network Output", color="red", alpha=0.8)
plt.axvline(x=10000, color="gray", linestyle=":", label="Error-driven (Weights Fixed)")
plt.axvline(x=15000, color="green", linestyle="-.", label="Free-running (No Error Input)")
plt.title(f"Sine Wave Generation Task (3-Phase Simulation - Epoch {num_epochs})")
plt.xlabel("Time (ms)")
plt.ylabel("Signal Amplitude")
plt.legend(loc="upper right")
plt.grid(True)

# ② DTW距離の推移
plt.subplot(3, 1, 2)
plt.plot(history['epoch'], history['err_dtw'], label="Error-driven DTW", color="orange", marker="o")
plt.plot(history['epoch'], history['free_dtw'], label="Free-running DTW", color="green", marker="s")
plt.title("DTW Distance Progress across Epochs")
plt.xlabel("Epoch")
plt.ylabel("DTW Distance")
plt.legend()
plt.grid(True)

# ③ 平均発火率の推移
plt.subplot(3, 1, 3)
plt.plot(history['epoch'], history['train_fr'], label="Training FR", color="blue")
plt.plot(history['epoch'], history['err_fr'], label="Error-driven FR", color="orange")
plt.plot(history['epoch'], history['free_fr'], label="Free-running FR", color="green")
plt.axhline(y=10.0, color="red", linestyle="--", label="Target FR (10 Hz)")
plt.title("Average Firing Rate (FR) Progress")
plt.xlabel("Epoch")
plt.ylabel("Firing Rate (Hz)")
plt.legend()
plt.grid(True)

plt.tight_layout()

# 結果画像の保存
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plot_filename = f"sine_wave_3phase_result_{current_time}.png"
plt.savefig(plot_filename)
print(f"Evaluation plot successfully saved as '{plot_filename}'")
