import numpy as np
import datetime
import matplotlib
# 画面出力のないCUI/CLI環境でエラーを防ぐためのバックエンド設定
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

class OUNoiseGenerator:
    """
    PredictiveEPropNetから抽出したOUノイズジェネレータ
    """
    def __init__(self, n_neurons, tau_s=250.0, sigma_s=0.02, mu_s=0.0):
        self.n_neurons = n_neurons
        self.tau_s = tau_s
        self.sigma_s = sigma_s
        self.mu_s = mu_s
        self.s = np.zeros(self.n_neurons)

    def generate_ou_noise(self, dt=1.0):
        # 元コードの数式に忠実な実装
        decay = np.exp(-dt / self.tau_s)
        self.s = self.mu_s + (self.s - self.mu_s) * decay + \
                 self.sigma_s * np.sqrt(1.0 - decay ** 2) * np.random.randn(self.n_neurons)
        return self.s.copy()

def main():
    # シミュレーションパラメータ
    n_neurons = 1        # プロットを見やすくするため3サンプル
    timesteps = 20000     # シミュレーションのステップ数
    dt = 1.0             # タイムステップ (ms)

    # ジェネレータのインスタンス化
    generator = OUNoiseGenerator(
        n_neurons=n_neurons, 
        tau_s=250.0, 
        sigma_s=1.0, 
        mu_s=0.0
    )

    # 結果を格納する配列
    noise_history = np.zeros((timesteps, n_neurons))

    # ノイズ生成ループ
    for t in range(timesteps):
        noise_history[t] = generator.generate_ou_noise(dt=dt)

    # 結果のプロット
    plt.figure(figsize=(12, 6))
    
    # 各ニューロンのノイズをプロット
    for i in range(n_neurons):
        plt.plot(noise_history[:, i], label=f'Neuron {i+1}', alpha=0.8, linewidth=1.5)

    # 見栄えの調整
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # 警告を回避するために rf (raw f-string) を使用
    plt.title(rf'Ornstein-Uhlenbeck Noise ($\mu={generator.mu_s}$, $\sigma={generator.sigma_s}$, $\tau={generator.tau_s}$)', fontsize=14)
    plt.xlabel('Time steps (t)', fontsize=12)
    plt.ylabel('Noise Amplitude (s)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 実行日時を取得してユニークなファイル名を生成 (YYYYMMDD_HHMMSS 形式)
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'ou_output/ou_noise_plot_{current_time}.png'
    
    # 画像ファイルとして保存
    plt.savefig(output_filename, dpi=150)
    plt.close() # メモリ解放
    
    print(f"シミュレーション完了: 結果を '{output_filename}' として保存しました。")

if __name__ == "__main__":
    main()