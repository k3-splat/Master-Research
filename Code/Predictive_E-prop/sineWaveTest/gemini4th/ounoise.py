import numpy as np
import matplotlib.pyplot as plt

# オルナシュテイン-ウーレンベック過程のパラメータ
theta = 0.1
mu = 0.5
sigma = 0.2

# 数値解法（オイラー・マリュマ法）
def ornstein_uhlenbeck_simulation(dt, T):
    t = np.arange(0, T, dt)
    n = len(t)
    X = np.zeros(n)

    for i in range(1, n):
        dW = np.random.normal(0, np.sqrt(dt))
        X[i] = X[i - 1] + theta * (mu - X[i - 1]) * dt + sigma * dW

    return t, X

# パラメータの設定
dt = 0.01  # タイムステップ
T = 1.0   # シミュレーションの総時間

# 数値解の計算
t, X = ornstein_uhlenbeck_simulation(dt, T)

# 結果のプロット
plt.plot(t, X)
plt.title("Ornstein-Uhlenbeck Process Simulation")
plt.xlabel("Time")
plt.ylabel("Value")

# show()の代わりにsavefig()を使用してファイルに保存する
# dpi=300で高画質化、bbox_inches='tight'で余白を自動調整します
plt.savefig("ou_process_simulation.png", dpi=300, bbox_inches='tight')

print("画像を 'ou_process_simulation.png' として保存しました。")