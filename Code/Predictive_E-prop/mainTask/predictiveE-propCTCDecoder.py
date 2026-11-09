import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Dict

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# ==============================================================================
# 1. TIMIT 音素定義および 61 -> 39 縮約マッピングテーブル
# ==============================================================================
# 61音素リスト (Index 0 は CTC [blank] トークンとして予約)
TIMIT_61_PHONEMES = [
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b',
    'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en',
    'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv',
    'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng',
    'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh',
    't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh'
]

# 61音素 -> 39音素縮約テーブル (Graves et al., 2006 / e-prop論文準拠)
PHONEME_61_TO_39_MAP = {
    'ao': 'aa',
    'ax': 'ah', 'ax-h': 'ah',
    'axr': 'er',
    'hv': 'hh',
    'ix': 'ih',
    'el': 'l',
    'em': 'm',
    'en': 'n', 'nx': 'n',
    'eng': 'ng',
    'zh': 'sh',
    'ux': 'uw',
    # 無音・閉鎖区間の縮約 (全て 'sil' に統合)
    'pcl': 'sil', 'tcl': 'sil', 'kcl': 'sil', 'bcl': 'sil', 'dcl': 'sil', 'gcl': 'sil',
    'h#': 'sil', 'pau': 'sil', 'epi': 'sil', 'q': 'sil'
}

# 39音素リスト
REDUCED_39_PHONEMES = [
    'sil', 'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh',
    'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh',
    'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's',
    'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z'
]

# ID変換テーブルの構築
PHN61_TO_ID = {phn: i + 1 for i, phn in enumerate(TIMIT_61_PHONEMES)}  # 1-indexed (0 is blank)
ID_TO_PHN61 = {i + 1: phn for i, phn in enumerate(TIMIT_61_PHONEMES)}
PHN39_TO_ID = {phn: i for i, phn in enumerate(REDUCED_39_PHONEMES)}


def build_reduction_mapping_tensor(device=device) -> torch.Tensor:
    """
    61クラスのインデックス (1〜61) を 39クラスのインデックス (0〜38) に変換するルックアップテンソル
    """
    mapping = torch.zeros(len(TIMIT_61_PHONEMES) + 1, dtype=torch.long, device=device)
    mapping[0] = -1  # blank
    for idx_61 in range(1, len(TIMIT_61_PHONEMES) + 1):
        phn_61 = ID_TO_PHN61[idx_61]
        target_phn_39 = PHONEME_61_TO_39_MAP.get(phn_61, phn_61)
        mapping[idx_61] = PHN39_TO_ID[target_phn_39]
    return mapping


# ==============================================================================
# 2. 既存Predictive E-prop RSNNのニューロン層定義 (Stage 1)
# ==============================================================================
@dataclass
class Config:
    time_step_dt: float = 1.0
    eta_rec: float = 0.0004
    eta_out: float = 0.0004
    eta_dec: float = 0.001
    lambda_w: float = 5e-6
    lambda_reg: float = 2.0
    t_delay: int = 10
    f_star: float = 0.01

    n_inputs: int = 123     # Filterbank 41チャネル × 3 (動的特徴量含む)
    n_lif: int = 100
    n_alif: int = 200
    n_outputs: int = 123    # 自己教師あり予測対象 (入力の再構成)
    n_classes_ctc: int = 62 # 61音素 + 1 blank (ID: 0)
    density: float = 0.3

    gamma_d: float = 0.3
    v_th: float = 0.6
    I_bias: float = 0.02
    tau_mem: float = 20.0
    tau_alif: float = 2000.0
    beta: float = 0.3
    tau_r: float = 50.0
    tau_d: float = 125.0
    t_ref: float = 5.0
    g: float = 150.0

    n_repeat: int = 5      # 1音響フレーム(10ms)に対するSNN提示ステップ数 (e-prop TIMIT標準)
    beam_width: int = 100

    @property
    def n_neurons(self) -> int:
        return self.n_lif + self.n_alif


class BaseNeuronGroup:
    def __init__(self, n_neurons: int, cfg: Config, device=device):
        self.n = n_neurons
        self.cfg = cfg
        self.device = device
        self.alpha = float(np.exp(-cfg.time_step_dt / cfg.tau_mem))
        self.decay_r = float(np.exp(-cfg.time_step_dt / cfg.tau_r))
        self.decay_d = float(np.exp(-cfg.time_step_dt / cfg.tau_d))
        self.c_rd = 1.0 / (cfg.tau_r * cfg.tau_d)

    def reset_state(self, batch_size: int):
        self.batch_size = batch_size
        self.v = torch.zeros((batch_size, self.n), device=self.device, dtype=torch.float32)
        self.z = torch.zeros((batch_size, self.n), device=self.device, dtype=torch.float32)
        self.z_bar = torch.zeros((batch_size, self.n), device=self.device, dtype=torch.float32)
        self.z_bar_bar = torch.zeros((batch_size, self.n), device=self.device, dtype=torch.float32)
        self.q = torch.full((batch_size, self.n), self.c_rd, device=self.device, dtype=torch.float32)
        self.psi = torch.zeros((batch_size, self.n), device=self.device, dtype=torch.float32)
        self.b = torch.full((batch_size, self.n), self.cfg.v_th, device=self.device, dtype=torch.float32)
        self.ref_counts = torch.zeros((batch_size, self.n), device=self.device, dtype=torch.float32)

    def _update_spikes_and_filters(self, v_next_raw: torch.Tensor, b_next: torch.Tensor):
        self.psi = (self.cfg.gamma_d / self.cfg.v_th) * torch.clamp(1.0 - torch.abs((v_next_raw - b_next) / self.cfg.v_th), min=0.0)
        is_currently_refractory = self.ref_counts > 0
        self.psi[is_currently_refractory] = 0.0
        spike_cond = (v_next_raw > b_next) & (~is_currently_refractory)
        self.z = spike_cond.float()

        v_next = torch.where(is_currently_refractory, self.v, v_next_raw - self.z * b_next)
        self.ref_counts = torch.clamp(self.ref_counts - self.cfg.time_step_dt, min=0.0)
        self.ref_counts = torch.where(self.z > 0, torch.tensor(self.cfg.t_ref, device=self.device, dtype=torch.float32), self.ref_counts)

        self.z_bar_bar = self.decay_d * self.z_bar_bar + self.q
        self.q = self.decay_r * self.q + self.c_rd * self.z
        self.z_bar = self.alpha * self.z_bar + self.z

        self.v = v_next
        self.b = b_next


class LIFGroup(BaseNeuronGroup):
    def __init__(self, n_neurons: int, cfg: Config, **kwargs):
        super().__init__(n_neurons, cfg, **kwargs)
        self.beta = 0.0

    def step(self, input_current: torch.Tensor):
        v_next_raw = self.alpha * self.v + input_current
        b_next = torch.full((self.batch_size, self.n), self.cfg.v_th, device=self.device, dtype=torch.float32)
        self._update_spikes_and_filters(v_next_raw, b_next)


class ALIFGroup(BaseNeuronGroup):
    def __init__(self, n_neurons: int, cfg: Config, **kwargs):
        super().__init__(n_neurons, cfg, **kwargs)
        self.beta = cfg.beta
        self.rho = float(np.exp(-cfg.time_step_dt / cfg.tau_alif))

    def reset_state(self, batch_size: int):
        super().reset_state(batch_size)
        self.a = torch.zeros((batch_size, self.n), device=self.device, dtype=torch.float32)

    def step(self, input_current: torch.Tensor):
        self.a = self.rho * self.a + self.z
        v_next_raw = self.alpha * self.v + input_current
        b_next = self.cfg.v_th + self.beta * self.a
        self._update_spikes_and_filters(v_next_raw, b_next)


class PredictiveEPropRSNN(nn.Module):
    """
    Stage 1: 自己教師あり Predictive E-prop RSNN
    入力 d^t = y^t - x^t を受け取り、内部表現 z_double_hat を生成する
    """
    def __init__(self, cfg: Config, device=device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.lif = LIFGroup(cfg.n_lif, cfg=cfg, device=device)
        self.alif = ALIFGroup(cfg.n_alif, cfg=cfg, device=device)
        self.neurons = [self.lif, self.alif]

        # Stage 1 重みパラメータ
        self.w_in = nn.Parameter(torch.randn(cfg.n_neurons, cfg.n_inputs, device=device) / np.sqrt(cfg.g), requires_grad=False)
        self.w_fb = nn.Parameter(torch.randn(cfg.n_neurons, cfg.n_outputs, device=device) / np.sqrt(cfg.g), requires_grad=False)
        
        mask = (torch.rand(cfg.n_neurons, cfg.n_neurons, device=device) < cfg.density).float()
        mask.fill_diagonal_(0.0)
        self.register_buffer("w_rec_mask", mask)
        w_rec_init = (torch.randn(cfg.n_neurons, cfg.n_neurons, device=device) / np.sqrt(cfg.n_neurons)) * mask
        self.w_rec = nn.Parameter(w_rec_init, requires_grad=False)
        self.w_out = nn.Parameter(torch.randn(cfg.n_outputs, cfg.n_neurons, device=device) / np.sqrt(cfg.n_neurons), requires_grad=False)

    def reset_state(self, batch_size: int):
        self.lif.reset_state(batch_size)
        self.alif.reset_state(batch_size)

    def forward_step(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1ステップ (1 ms) のRSNNフォワード処理
        戻り値: (z_bar_bar_t, y_t)
        """
        all_z_bar_bar = torch.cat([self.lif.z_bar_bar, self.alif.z_bar_bar], dim=1)
        y_t = all_z_bar_bar @ self.w_out.t()
        d_t_pred = y_t - x_t

        i_rec = all_z_bar_bar @ self.w_rec.t()
        i_in = d_t_pred @ self.w_in.t()
        i_fb = y_t @ self.w_fb.t()
        total_current = i_in + i_rec + i_fb + self.cfg.I_bias

        self.lif.step(total_current[:, :self.cfg.n_lif])
        self.alif.step(total_current[:, self.cfg.n_lif:])

        all_z_bar_bar_next = torch.cat([self.lif.z_bar_bar, self.alif.z_bar_bar], dim=1)
        return all_z_bar_bar_next, y_t

    def extract_features(self, x_seq: torch.Tensor, seq_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        時系列入力 x_seq: [B, T_raw, n_inputs] から平滑化表現 \bar{\bar{z}} を抽出し、
        SNN時間ステップ (n_repeat) をフレームレートに平均プーリングしてダウンサンプリングする
        戻り値: (features [B, T_frames, n_rec], frame_lengths [B])
        """
        B, T_raw, _ = x_seq.shape
        self.reset_state(B)

        z_bar_bar_list = []
        for t in range(T_raw):
            z_bb, _ = self.forward_step(x_seq[:, t, :])
            z_bar_bar_list.append(z_bb)

        # [B, T_raw, n_rec]
        z_bar_bar_seq = torch.stack(z_bar_bar_list, dim=1)

        # n_repeat ごとにダウンサンプリング (平均プーリング)
        if self.cfg.n_repeat > 1:
            T_frames = T_raw // self.cfg.n_repeat
            z_reshaped = z_bar_bar_seq[:, :T_frames * self.cfg.n_repeat, :].view(B, T_frames, self.cfg.n_repeat, self.cfg.n_neurons)
            features = z_reshaped.mean(dim=2)  # [B, T_frames, n_rec]
            frame_lengths = torch.clamp(seq_lengths // self.cfg.n_repeat, min=1)
        else:
            features = z_bar_bar_seq
            frame_lengths = seq_lengths

        return features, frame_lengths


# ==============================================================================
# 3. Stage 2: CTCデコーダ (nn.Module)
# ==============================================================================
class CTCDecoder(nn.Module):
    """
    Stage 2: 平滑化スパイク表現 \bar{\bar{z}}^t を音素Logitsへ射影する線形デコーダ
    """
    def __init__(self, n_rec: int, n_classes: int = 62):
        super().__init__()
        self.n_rec = n_rec
        self.n_classes = n_classes  # 61 classes + 1 blank (blank = 0)
        self.linear = nn.Linear(n_rec, n_classes)
        
        # 重みの初期化 (Xavier / e-prop標準の1/sqrt(fan_in))
        nn.init.normal_(self.linear.weight, mean=0.0, std=1.0 / math.sqrt(n_rec))
        nn.init.zeros_(self.linear.bias)

    def forward(self, z_double_hat: torch.Tensor) -> torch.Tensor:
        """
        z_double_hat: [B, T, n_rec]
        戻り値: log_probs [T, B, n_classes] (PyTorch nn.CTCLoss 標準形式)
        """
        logits = self.linear(z_double_hat)             # [B, T, n_classes]
        log_probs = F.log_softmax(logits, dim=-1)      # [B, T, n_classes]
        return log_probs.transpose(0, 1)               # [T, B, n_classes]


# ==============================================================================
# 4. CTC Loss 計算モジュール
# ==============================================================================
class CTCLossCalculator:
    def __init__(self, blank_idx: int = 0, zero_infinity: bool = True):
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=zero_infinity)

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        log_probs: [T, B, C]
        targets: 1D Tensor [sum(target_lengths)] もしくは 2D Tensor [B, L_max]
        input_lengths: [B]
        target_lengths: [B]
        """
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        return loss


# ==============================================================================
# 5. ビームサーチデコードおよび PER (Phoneme Error Rate) 評価
# ==============================================================================
def ctc_beam_search_decoder(
    log_probs: torch.Tensor,
    seq_len: int,
    beam_width: int = 100,
    blank_idx: int = 0
) -> List[int]:
    """
    プレフィックスツリーを探索するCTCビームサーチアルゴリズム
    log_probs: [T, C] (単一サンプル、対数確率)
    戻り値: 最も確率の高い音素ID系列 (リスト)
    """
    T, C = log_probs.shape
    # ビームの状態: tuple(prefix) -> (log_p_blank, log_p_non_blank)
    beam: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, -float('inf'))}

    for t in range(seq_len):
        lp_t = log_probs[t]
        new_beam: Dict[Tuple[int, ...], Tuple[float, float]] = {}

        def log_sum_exp(a: float, b: float) -> float:
            if a == -float('inf'): return b
            if b == -float('inf'): return a
            return max(a, b) + math.log1p(math.exp(-abs(a - b)))

        for prefix, (p_b, p_nb) in beam.items():
            p_total = log_sum_exp(p_b, p_nb)

            # 1. Blank 遷移
            lp_blank = float(lp_t[blank_idx])
            nb_b, nb_nb = new_beam.get(prefix, (-float('inf'), -float('inf')))
            new_beam[prefix] = (log_sum_exp(nb_b, p_total + lp_blank), nb_nb)

            # 2. Non-Blank 遷移
            for c in range(C):
                if c == blank_idx:
                    continue
                lp_c = float(lp_t[c])
                if len(prefix) > 0 and prefix[-1] == c:
                    # 同一文字が連続する場合: Blank経由でないと伸長しない
                    new_prefix = prefix + (c,)
                    nb_b_ext, nb_nb_ext = new_beam.get(new_prefix, (-float('inf'), -float('inf')))
                    new_beam[new_prefix] = (nb_b_ext, log_sum_exp(nb_nb_ext, p_b + lp_c))

                    # 直前のNon-blankからの遷移は同一プレフィックスにマージ
                    nb_b_curr, nb_nb_curr = new_beam.get(prefix, (-float('inf'), -float('inf')))
                    new_beam[prefix] = (nb_b_curr, log_sum_exp(nb_nb_curr, p_nb + lp_c))
                else:
                    new_prefix = prefix + (c,)
                    nb_b_ext, nb_nb_ext = new_beam.get(new_prefix, (-float('inf'), -float('inf')))
                    new_beam[new_prefix] = (nb_b_ext, log_sum_exp(nb_nb_ext, p_total + lp_c))

        # ビーム幅でプルーニング
        beam = dict(
            sorted(
                new_beam.items(),
                key=lambda item: log_sum_exp(item[1][0], item[1][1]),
                reverse=True
            )[:beam_width]
        )

    best_prefix = max(beam.keys(), key=lambda p: log_sum_exp(beam[p][0], beam[p][1]))
    return list(best_prefix)


def levenshtein_distance(seq1: List[int], seq2: List[int]) -> Tuple[int, int, int, int]:
    """
    2つの系列間の編集距離および置換(S)、挿入(I)、削除(D)の数を計算
    戻り値: (total_errors, substitutions, insertions, deletions)
    """
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    ops = np.zeros((n + 1, m + 1, 3), dtype=int)  # S, I, D

    for i in range(n + 1):
        dp[i][0] = i
        ops[i][0] = [0, 0, i]
    for j in range(m + 1):
        dp[0][j] = j
        ops[0][j] = [0, j, 0]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = ops[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1] + 1
                ins = dp[i][j - 1] + 1
                dlt = dp[i - 1][j] + 1

                min_val = min(sub, ins, dlt)
                dp[i][j] = min_val
                if min_val == sub:
                    ops[i][j] = ops[i - 1][j - 1] + [1, 0, 0]
                elif min_val == ins:
                    ops[i][j] = ops[i][j - 1] + [0, 1, 0]
                else:
                    ops[i][j] = ops[i - 1][j] + [0, 0, 1]

    s, i_err, d = ops[n][m]
    return dp[n][m], int(s), int(i_err), int(d)


@torch.no_grad()
def evaluate_per(
    decoder: CTCDecoder,
    rsnn: PredictiveEPropRSNN,
    x_batch: torch.Tensor,
    seq_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    reduction_map: torch.Tensor,
    beam_width: int = 100
) -> Dict[str, float]:
    """
    音素エラー率 (PER) を計算
    - 61クラス音素系列をビームサーチデコード
    - 61 -> 39 縮約マッピングテーブルを適用
    - Levenshtein 距離から PER, 置換率, 挿入率, 削除率を算出
    """
    decoder.eval()
    rsnn.eval()

    # Stage 1 特徴抽出
    features, frame_lengths = rsnn.extract_features(x_batch, seq_lengths)
    # Stage 2 デコード [T, B, 62]
    log_probs = decoder(features)

    B = x_batch.shape[0]
    total_distance = 0
    total_ref_len = 0
    total_sub = 0
    total_ins = 0
    total_del = 0

    log_probs_cpu = log_probs.cpu()
    frame_lengths_cpu = frame_lengths.cpu().tolist()
    target_lengths_cpu = target_lengths.cpu().tolist()
    targets_cpu = targets.cpu()
    reduction_map_cpu = reduction_map.cpu().numpy()

    target_offset = 0
    for b in range(B):
        flen = frame_lengths_cpu[b]
        tlen = target_lengths_cpu[b]
        ref_seq_61 = targets_cpu[target_offset:target_offset + tlen].tolist()
        target_offset += tlen

        # ビームサーチデコード (61クラス)
        pred_seq_61 = ctc_beam_search_decoder(log_probs_cpu[:, b, :], seq_len=flen, beam_width=beam_width)

        # 61音素 -> 39音素 縮約マッピングの適用
        pred_seq_39 = [int(reduction_map_cpu[p]) for p in pred_seq_61 if p > 0]
        ref_seq_39 = [int(reduction_map_cpu[r]) for r in ref_seq_61 if r > 0]

        # 編集距離の計算
        dist, s, ins, d = levenshtein_distance(ref_seq_39, pred_seq_39)
        total_distance += dist
        total_sub += s
        total_ins += ins
        total_del += d
        total_ref_len += len(ref_seq_39)

    per = (total_distance / max(total_ref_len, 1)) * 100.0
    sub_rate = (total_sub / max(total_ref_len, 1)) * 100.0
    ins_rate = (total_ins / max(total_ref_len, 1)) * 100.0
    del_rate = (total_del / max(total_ref_len, 1)) * 100.0

    return {
        "PER": per,
        "Sub_Rate": sub_rate,
        "Ins_Rate": ins_rate,
        "Del_Rate": del_rate,
        "Total_Errors": total_distance,
        "Ref_Phonemes": total_ref_len
    }


# ==============================================================================
# 6. メイン実行用サンプルコード
# ==============================================================================
if __name__ == '__main__':
    print("=== Predictive E-prop CTC Decoder (Stage 2) Pipeline Test ===")
    
    cfg = Config()
    print(f"Device: {device}")
    print(f"RSNN Size: {cfg.n_neurons} (LIF: {cfg.n_lif}, ALIF: {cfg.n_alif})")
    print(f"Inputs: {cfg.n_inputs} dims | Decoder Classes: {cfg.n_classes_ctc} (61 Phns + 1 Blank)")
    print(f"Temporal Repeat factor: {cfg.n_repeat} (e-prop TIMIT specification)")

    # 1. モデルのインスタンス化
    rsnn = PredictiveEPropRSNN(cfg, device=device)
    decoder = CTCDecoder(n_rec=cfg.n_neurons, n_classes=cfg.n_classes_ctc).to(device)
    ctc_calculator = CTCLossCalculator(blank_idx=0, zero_infinity=True)
    reduction_map = build_reduction_mapping_tensor(device=device)

    # 2. 重みのフリーズ確認: Stage 1 (RSNN) はフリーズ、Stage 2 (Decoder) のみ学習対象
    for param in rsnn.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = True

    trainable_params = list(filter(lambda p: p.requires_grad, decoder.parameters()))
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.eta_dec, eps=1e-5)
    print(f"Stage 1 RSNN frozen parameters: {sum(p.numel() for p in rsnn.parameters())}")
    print(f"Stage 2 Decoder trainable parameters: {sum(p.numel() for p in trainable_params)}")

    # 3. ダミーのバッチデータ生成 (Filterbank特徴量 + 音素ラベル)
    # バッチサイズ 4、可変長シーケンス
    batch_size = 4
    num_frames = [40, 55, 30, 48]  # 音響特徴量フレーム数 (10 ms単位)
    max_frames = max(num_frames)
    max_snn_steps = max_frames * cfg.n_repeat

    # SNN入力 [B, max_snn_steps, n_inputs] (5ms提示を再現)
    raw_audio_features = torch.randn(batch_size, max_frames, cfg.n_inputs, device=device)
    snn_input = torch.repeat_interleave(raw_audio_features, repeats=cfg.n_repeat, dim=1)
    snn_lengths = torch.tensor([f * cfg.n_repeat for f in num_frames], device=device, dtype=torch.long)

    # ターゲット音素系列 (1〜61のID、可変長)
    target_lengths_list = [12, 18, 9, 15]
    target_lengths = torch.tensor(target_lengths_list, device=device, dtype=torch.long)
    targets_list = []
    for tlen in target_lengths_list:
        # ランダムな61音素ID (1〜61)
        targets_list.append(torch.randint(low=1, high=62, size=(tlen,), device=device, dtype=torch.long))
    targets = torch.cat(targets_list)

    print("\n--- Running Forward Step ---")
    # 4. Stage 1: 特徴抽出 (勾配計算不要)
    with torch.no_grad():
        features, frame_lengths = rsnn.extract_features(snn_input, snn_lengths)
    print(f"Extracted SNN Features Shape: {features.shape} [Batch, Frames, Neurons]")
    print(f"Frame lengths after downsampling: {frame_lengths.tolist()}")

    # 5. Stage 2: デコーダ順伝播
    decoder.train()
    optimizer.zero_grad()
    log_probs = decoder(features)  # [T, B, 62]
    print(f"Log Probs Shape: {log_probs.shape} [Time, Batch, Classes]")

    # 6. CTC Loss 計算と誤差逆伝播
    loss = ctc_calculator.compute_loss(log_probs, targets, frame_lengths, target_lengths)
    print(f"Initial CTC Loss: {loss.item():.4f}")

    loss.backward()
    optimizer.step()
    print("Backward pass & Optimizer step completed successfully.")

    # 7. PER 評価
    print("\n--- Running Evaluation (PER with 61 -> 39 mapping) ---")
    metrics = evaluate_per(
        decoder=decoder,
        rsnn=rsnn,
        x_batch=snn_input,
        seq_lengths=snn_lengths,
        targets=targets,
        target_lengths=target_lengths,
        reduction_map=reduction_map,
        beam_width=cfg.beam_width
    )

    print(f"Evaluation Results:")
    print(f"  Phoneme Error Rate (PER) : {metrics['PER']:.2f}%")
    print(f"  Substitutions Rate       : {metrics['Sub_Rate']:.2f}%")
    print(f"  Insertions Rate          : {metrics['Ins_Rate']:.2f}%")
    print(f"  Deletions Rate           : {metrics['Del_Rate']:.2f}%")
    print(f"  Total Reference Phones   : {metrics['Ref_Phonemes']}")
    print("\nTest finished successfully.")