import torch
import datetime

class SineWaveGenerator:
    """
    Predictive E-prop実験用のSine波生成モジュール
    """
    def __init__(self, amplitude=0.4, period=1000.0, phase=0.0, offset=0.0, dt=1.0):
        """
        パラメータの初期化
        Args:
            amplitude (float): 振幅 (A)
            period (float): 周期 (T) [ms]
            phase (float): 位相 (phi)
            offset (float): オフセット (c)
            dt (float): タイムステップ [ms]
        """
        self.amplitude = amplitude
        self.period = period
        self.phase = phase
        self.offset = offset
        self.dt = dt

    def generate(self, total_time):
        """
        指定された合計時間 (ms) に対するSine波の時系列データを生成する
        
        Args:
            total_time (float or int): 生成する総時間 (ms)
            
        Returns:
            torch.Tensor: 生成されたSine波の時系列データ (形状: [time_steps, 1])
        """
        # タイムステップの配列を生成
        time_steps = int(total_time / self.dt)
        t = torch.arange(0, time_steps, dtype=torch.float32) * self.dt
        
        # Sine波の数式: x(t) = A * sin( (2 * pi / T) * t + phi ) + c
        signal = self.amplitude * torch.sin((2 * torch.pi / self.period) * t + self.phase) + self.offset
        
        # ネットワークへの入力として扱いやすいように形状を [time_steps, 1] に変更
        return signal.unsqueeze(1)

# ==========================================
# テスト実行コード (単体テスト用)
# ==========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 論文に記載されているSine波のデフォルトパラメータで初期化
    generator = SineWaveGenerator(
        amplitude=0.4, 
        period=1000.0, 
        phase=0.0, 
        offset=0.0, 
        dt=1.0
    )
    
    # エポック全体の長さに相当する20000ms (20秒) 分のデータを生成
    total_duration = 20000
    target_signal = generator.generate(total_duration)
    
    print(f"Generated signal shape: {target_signal.shape}")
    
    # 最初の2000ms (2秒, 2周期分) だけプロットして波形を確認
    t_plot = torch.arange(0, 2000, generator.dt).numpy()
    plt.plot(t_plot, target_signal.numpy()[:2000])
    plt.title("Generated Sine Wave Target")
    plt.xlabel("Time (ms)")
    plt.ylabel("Signal Value")
    plt.grid(True)
    
    # 日時を取得してユニークなファイル名を生成し、画像を保存
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sine_wave_target_{current_time}.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    
    # リソース解放
    plt.close()