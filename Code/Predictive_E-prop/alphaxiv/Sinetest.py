# Sinetest.py
#
# Predictive E-propによる正弦波予測実験
#
# 実行:
#   python Sinetest.py
#
# 依存:
#   numpy
#   matplotlib
#
# 短時間で試す場合:
#   python Sinetest.py --epochs 5 --train-ms 2000 --error-ms 1000 --free-ms 1000
#
# 論文設定に近づける場合:
#   python Sinetest.py --epochs 50 --train-ms 10000 --error-ms 5000 --free-ms 5000

import argparse
import numpy as np
import matplotlib.pyplot as plt

from Network import PredictiveEpropNetwork


def make_sine(
    duration_ms,
    dt_ms=1.0,
    amplitude=0.4,
    period_ms=1000.0,
    phase=0.0,
    offset=0.0,
):
    """
    正弦波を生成する。

    x(t) = A sin(2*pi*t/T + phase) + offset
    """
    n_steps = int(round(duration_ms / dt_ms))
    t_ms = np.arange(n_steps) * dt_ms

    x = (
        amplitude
        * np.sin(2.0 * np.pi * t_ms / period_ms + phase)
        + offset
    )

    return t_ms, x[:, None]


def dtw_distance(x, y):
    """
    1次元時系列に対するDynamic Time Warping距離。
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    n = len(x)
    m = len(y)

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distance = abs(x[i - 1] - y[j - 1])

            cost[i, j] = distance + min(
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1],
            )

    return cost[n, m]


def run_experiment(args):
    dt_ms = args.dt_ms

    # ------------------------------------------------------------
    # ネットワーク
    # ------------------------------------------------------------

    network = PredictiveEpropNetwork(
        n_lif=args.n_lif,
        n_alif=args.n_alif,
        input_dim=1,
        output_dim=1,
        dt_ms=dt_ms,
        input_gain=args.input_gain,
        recurrent_sparsity=args.sparsity,
        eta_rec=args.eta_rec,
        eta_out=args.eta_out,
        weight_decay=args.weight_decay,
        homeo_strength=args.homeo_strength,
        target_rate_hz=10.0,
        homeo_interval_ms=10.0,
        tau_mem_ms=20.0,
        tau_alif_ms=2000.0,
        tau_r_ms=50.0,
        tau_d_ms=125.0,
        threshold=0.6,
        bias_current=0.02,
        ou_tau_ms=250.0,
        ou_sigma=1.0,
        ou_mu=0.0,
        surrogate_gamma=0.3,
        refractory_ms=5.0,
        seed=args.seed,
    )

    # ------------------------------------------------------------
    # 各phaseの目標正弦波
    # ------------------------------------------------------------

    _, train_target = make_sine(
        args.train_ms,
        dt_ms=dt_ms,
        amplitude=0.4,
        period_ms=1000.0,
        phase=0.0,
        offset=0.0,
    )

    _, error_target = make_sine(
        args.error_ms,
        dt_ms=dt_ms,
        amplitude=0.4,
        period_ms=1000.0,
        phase=0.0,
        offset=0.0,
    )

    _, free_target = make_sine(
        args.free_ms,
        dt_ms=dt_ms,
        amplitude=0.4,
        period_ms=1000.0,
        phase=0.0,
        offset=0.0,
    )

    # ------------------------------------------------------------
    # 学習
    # ------------------------------------------------------------

    history = []

    print("Start training")

    for epoch in range(args.epochs):
        # --------------------------------------------------------
        # 1. Training phase
        #    誤差入力あり、重み更新あり
        # --------------------------------------------------------

        train_output, train_spikes = network.run_phase(
            train_target,
            learn=True,
            error_input_enabled=True,
            reset_state=True,
        )

        # --------------------------------------------------------
        # 2. Error-driven phase
        #    誤差入力あり、重み固定
        # --------------------------------------------------------

        error_output, error_spikes = network.run_phase(
            error_target,
            learn=False,
            error_input_enabled=True,
            reset_state=True,
        )

        # --------------------------------------------------------
        # 3. Free-running phase
        #    誤差入力なし、重み固定
        # --------------------------------------------------------

        free_output, free_spikes = network.run_phase(
            free_target,
            learn=False,
            error_input_enabled=False,
            reset_state=True,
        )

        # free-running phaseの最後の半分を評価
        start = len(free_target) // 2

        score = dtw_distance(
            free_target[start:],
            free_output[start:],
        )

        mse = np.mean(
            (free_target[start:] - free_output[start:]) ** 2
        )

        mean_rate_hz = (
            np.mean(free_spikes[start:])
            / dt_ms
            * 1000.0
        )

        history.append(
            {
                "epoch": epoch + 1,
                "dtw": score,
                "mse": mse,
                "mean_rate_hz": mean_rate_hz,
            }
        )

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"free DTW = {score:.4f} | "
            f"free MSE = {mse:.6f} | "
            f"rate = {mean_rate_hz:.2f} Hz"
        )

    # ------------------------------------------------------------
    # 最終結果を再計算
    # ------------------------------------------------------------

    final_train_output, final_train_spikes = network.run_phase(
        train_target,
        learn=False,
        error_input_enabled=True,
        reset_state=True,
    )

    final_error_output, final_error_spikes = network.run_phase(
        error_target,
        learn=False,
        error_input_enabled=True,
        reset_state=True,
    )

    final_free_output, final_free_spikes = network.run_phase(
        free_target,
        learn=False,
        error_input_enabled=False,
        reset_state=True,
    )

    # ------------------------------------------------------------
    # 結果を描画
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 12),
        constrained_layout=True,
    )

    # Training phase
    axes[0].plot(
        train_target[:, 0],
        label="target",
        linewidth=1.5,
    )
    axes[0].plot(
        final_train_output[:, 0],
        label="prediction",
        linewidth=1.0,
    )
    axes[0].set_title("Training phase")
    axes[0].set_ylabel("signal")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Error-driven phase
    axes[1].plot(
        error_target[:, 0],
        label="target",
        linewidth=1.5,
    )
    axes[1].plot(
        final_error_output[:, 0],
        label="prediction",
        linewidth=1.0,
    )
    axes[1].set_title("Error-driven phase")
    axes[1].set_ylabel("signal")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Free-running phase
    axes[2].plot(
        free_target[:, 0],
        label="target",
        linewidth=1.5,
    )
    axes[2].plot(
        final_free_output[:, 0],
        label="prediction",
        linewidth=1.0,
    )
    axes[2].set_title(
        "Free-running phase: prediction error input disabled"
    )
    axes[2].set_ylabel("signal")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    # 学習曲線
    epochs = [h["epoch"] for h in history]
    dtw_values = [h["dtw"] for h in history]
    mse_values = [h["mse"] for h in history]

    axes[3].plot(
        epochs,
        dtw_values,
        label="DTW",
        marker="o",
        markersize=3,
    )
    axes[3].plot(
        epochs,
        mse_values,
        label="MSE",
        marker="x",
        markersize=3,
    )
    axes[3].set_title("Learning curves")
    axes[3].set_xlabel("epoch")
    axes[3].set_ylabel("metric")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()

    # ネットワークサイズ
    parser.add_argument("--n-lif", type=int, default=100)
    parser.add_argument("--n-alif", type=int, default=200)

    # シミュレーション
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--train-ms", type=int, default=10000)
    parser.add_argument("--error-ms", type=int, default=5000)
    parser.add_argument("--free-ms", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=50)

    # ネットワークパラメータ
    parser.add_argument("--input-gain", type=float, default=150.0)
    parser.add_argument("--sparsity", type=float, default=0.99)
    parser.add_argument("--eta-rec", type=float, default=4e-4)
    parser.add_argument("--eta-out", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-6)
    parser.add_argument("--homeo-strength", type=float, default=2.0)

    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
