import torch
import torch.nn as nn
import scipy.io as sio
import time
from Network import PredictiveEPropRSNN

# ========================================================
# 1. データの読み込みと前処理
# ========================================================
# MATファイルの読み込み（パスはご自身の環境に合わせてください）
file_path = '/home/keitaro-sunagawa/Master-Research/Code/Encoder/se_saa08_binary_matrix.mat' 
mat_data = sio.loadmat(file_path)

# BAEからの出力は (620, 119999) 
binary_matrix = mat_data['binary_pattern']  

# PyTorchモデルの入力形状 (seq_len, batch_size, input_dim) に変換
# 転置して (119999, 620) にし、中間にbatch_size=1を追加 -> (119999, 1, 620)
target_spikes = torch.tensor(binary_matrix.T, dtype=torch.float32).unsqueeze(1)

# GPUが利用可能な場合はデバイスを設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_spikes = target_spikes.to(device)

# ========================================================
# 2. モデルとオプティマイザの初期化
# ========================================================
# フェーズ3までで定義したモデルをインスタンス化
model = PredictiveEPropRSNN(input_dim=620, n_lif=100, n_alif=200, dt=1.0).to(device)

# 論文のパラメータに基づくハイパーパラメータ設定
learning_rate = 0.0004 # 論文のサイン波再構成タスクの設定[3]
weight_decay = 5e-6    # 論文の重み減衰(lambda_W)の設定[3]

# Adamオプティマイザ (学習対象は W_rec と W_out のみ)
optimizer = torch.optim.Adam([model.W_rec, model.W_out], lr=learning_rate, weight_decay=weight_decay)

# ========================================================
# 3. 学習ループ (Epochs)
# ========================================================
num_epochs = 50 # 論文に基づく標準的なエポック数[4]
seq_len = target_spikes.size(0)

# ※重要：12万ステップのPythonループを一度に回すと時間がかかるため、
# 10,000ステップ（10秒分）ごとに区切って学習を進める「チャンク処理」を導入します
chunk_size = 10000 

print(f"Using device: {device}")
print(f"Total sequence length: {seq_len} steps")
print("Starting Training...")

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    epoch_loss = 0.0
    
    # チャンク（分割データ）ごとの処理
    for start_idx in range(0, seq_len, chunk_size):
        end_idx = min(start_idx + chunk_size, seq_len)
        x_chunk = target_spikes[start_idx:end_idx]
        
        # 勾配の初期化
        optimizer.zero_grad()
        
        # --- BPTTの無効化 ---
        # e-propは計算グラフを遡る必要がないため no_grad() で囲みます
        with torch.no_grad():
            
            # 順伝播の実行 (この内部のフェーズ3のコードで手動で .grad がセットされます)
            y_seq, z_seq, d_seq = model(x_chunk, free_running=False)
            
            # 平均二乗誤差（MSE）の計算 (ログ出力用、論文 Eq. 4に基づく[5])
            loss = 0.5 * torch.sum(d_seq ** 2) / x_chunk.size(0)
            epoch_loss += loss.item()
            
        # オプティマイザによる重みの更新 (手動セットされた .grad に基づく)
        optimizer.step()
        
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss (Pred Error): {epoch_loss:.4f} | Time: {epoch_time:.2f} s")

print("Training Completed!")