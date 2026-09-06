import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

# 既存の Network.py からモデルをインポート
from Network import PredictiveEPropRSNN

# ========================================================
# 1. 論文に基づくサイン波ターゲットデータの生成 (Eq. 14 / Table SI)
# ========================================================
A = 0.4
T_period = 1000.0  # 周期 1000 ms
dt = 1.0           # タイムステップ 1 ms
train_len = 10000  # 学習フェーズの長さ 10000 ms

t = np.arange(0, train_len, dt)
target_signal = A * np.sin(2 * np.pi * t / T_period)

# PyTorchのテンソル形状 (seq_len, batch_size, input_dim) に変換 (10000, 1, 1)
target_spikes = torch.tensor(target_signal, dtype=torch.float32).view(train_len, 1, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_spikes = target_spikes.to(device)

# ========================================================
# 2. モデルの初期化とサイン波用パラメータの上書き
# ========================================================
# サイン波は1次元なので input_dim=1 で初期化
model = PredictiveEPropRSNN(input_dim=1, n_lif=100, n_alif=200, dt=1.0).to(device)

# Table SI に基づき、ゲイン g をサイン波タスク用の 150 に設定し、固定重みを再初期化
model.g = 150.0
model.W_in = nn.Parameter(torch.randn(model.N, model.input_dim, device=device) / np.sqrt(model.g), requires_grad=False)
model.W_fb = nn.Parameter(torch.randn(model.N, model.input_dim, device=device) / np.sqrt(model.g), requires_grad=False)

learning_rate = 0.0004  # サイン波タスク用の学習率
weight_decay = 5e-6
optimizer = torch.optim.Adam([model.W_rec, model.W_out], lr=learning_rate, weight_decay=weight_decay)

# ========================================================
# 3. 学習ループ (BPTTを使わない e-prop の検証)
# ========================================================
num_epochs = 150

print(f"Using device: {device}")
print("Starting Sine Wave Training Task...")

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    
    optimizer.zero_grad()
    
    # BPTTを無効化し、Network.py内のe-propで勾配を計算
    with torch.no_grad():
        y_seq, z_seq, d_seq = model(target_spikes, free_running=False)
        # 予測誤差のMSEを計算 (論文 Eq. 4)
        loss = 0.5 * torch.sum(d_seq ** 2) / train_len
        
    optimizer.step()
    
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch [{epoch+1:2d}/{num_epochs}] | Loss (MSE): {loss.item():.6f} | Time: {epoch_time:.2f} s")

# ========================================================
# 4. 結果の可視化 (論文 Fig 1C, 2A の再現)
# ========================================================
# 学習完了後のネットワーク出力波形を取得
y_out = y_seq.squeeze().cpu().numpy()
target_out = target_signal

plt.figure(figsize=(10, 4))
# 論文の図に合わせて、学習期間全体（10000ms = 10秒分）をプロットするように変更
plt.plot(t, target_out, label="Target (Sine)", color="blue", linestyle="--")
plt.plot(t, y_out, label="Network Output", color="red", alpha=0.8)
plt.title("Sine Wave Reconstruction Task (10 seconds Training Phase)")
plt.xlabel("Time (ms)")
plt.ylabel("Signal Amplitude")
plt.legend(loc="upper right")
plt.grid(True)

# 現在の日時を取得してユニークなファイル名を作成 (例: sine_wave_result_20260729_225530.png)
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"sine_wave_result_{current_time}.png"

# CLI環境用に画像ファイルとして保存
plt.savefig(filename)
print(f"Plot saved as '{filename}'")