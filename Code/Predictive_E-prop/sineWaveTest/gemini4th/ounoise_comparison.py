import numpy as np
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. パラメータの設定 (PredictiveEPropNet側の設定を基準とする)
# ---------------------------------------------------------
dt = 1.0
T = 20000.0
timesteps = int(T / dt)

# 【厳密解法 (Exact) のパラメータ】
mu_s = 0.0
tau_s = 250.0
sigma_s = 1.0

# 【オイラー・マリュマ法 (Euler) のパラメータに変換】
mu = mu_s
theta = 1.0 / tau_s
sigma = sigma_s * np.sqrt(2.0 / tau_s)

# ---------------------------------------------------------
# 2. 共通の乱数（標準正規分布）を生成
# ---------------------------------------------------------
# 全く同じ波形を作るため、シードを固定して共通の乱数配列を準備
np.random.seed(42)
Z = np.random.randn(timesteps)

# ---------------------------------------------------------
# 3. シミュレーション実行
# ---------------------------------------------------------
t_array = np.arange(0, T, dt)
X_exact = np.zeros(timesteps)
X_euler = np.zeros(timesteps)

# 厳密解法用の減衰率
decay = np.exp(-dt / tau_s)

for i in range(1, timesteps):
    # ① 厳密解法 (PredictiveEPropNetの実装)
    X_exact[i] = mu_s + (X_exact[i-1] - mu_s) * decay + \
                 sigma_s * np.sqrt(1.0 - decay**2) * Z[i]
                 
    # ② オイラー・マリュマ法 (今回の実装)
    # dW = 共通乱数 Z[i] * sqrt(dt)
    dW = Z[i] * np.sqrt(dt)
    X_euler[i] = X_euler[i-1] + theta * (mu - X_euler[i-1]) * dt + sigma * dW

# ---------------------------------------------------------
# 4. 結果のプロット
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))

# 線を重ねて描画（Euler法を少し太く薄く描画し、Exact法を細く濃く描画）
plt.plot(t_array, X_euler, label='Euler-Maruyama Method (Approximate)', color='blue', alpha=0.5, linewidth=4)
plt.plot(t_array, X_exact, label='Exact Method (PredictiveEPropNet)', color='red', alpha=0.8, linewidth=1.5)

plt.axhline(mu_s, color='black', linestyle='--', linewidth=1, alpha=0.5)
plt.title('Comparison of OU Process Implementations (Same Conditions)', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# ユニークなファイル名で保存
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_filename = f"ou_output/ou_comparison_{current_time}.png"

plt.savefig(output_filename, dpi=150)
plt.close()

print(f"比較画像を '{output_filename}' として保存しました。")