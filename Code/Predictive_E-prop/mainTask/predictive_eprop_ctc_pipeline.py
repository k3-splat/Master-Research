import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import datetime
import os
import pickle
from dataclasses import dataclass, asdict

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")


# ==========================================
# 1. ハイパーパラメータ設定
# ==========================================
@dataclass
class Config:
    # タイムステップ (ms)
    time_step_dt: float = 1.0

    # 学習・最適化パラメータ
    eta_rec: float = 0.0004
    eta_out: float = 0.0004
    lambda_w: float = 5e-6
    lambda_reg: float = 2.0
    t_delay: int = 10
    f_star: float = 0.01

    # ネットワーク構造
    n_inputs: int = 123     # filter_banks.pickle の次元数 (41 * 3 = 123次元)
    n_outputs: int = 123    # 予測ターゲットも同じ次元数
    n_lif: int = 100
    n_alif: int = 200
    density: float = 0.2

    # CTCパラメータ
    n_phonemes_out: int = 62 # 61音素 + 1 blank
    blank_id: int = 0

    # ニューロン・シナプスダイナミクス
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

    # OUノイズパラメータ
    tau_s: float = 250.0
    sigma_s: float = 0.1
    mu_s: float = 0.0

    # 実行設定
    stage1_epochs: int = 5
    stage2_epochs: int = 10
    beam_width: int = 100
    
    # データパス設定
    features_pickle_path: str = "filter_banks.pickle"  # 音響特徴量 (123次元)
    targets_pickle_path: str = "reduced_phonems.pickle"  # 正解音素ラベル

    @property
    def n_neurons(self) -> int:
        return self.n_lif + self.n_alif


# ==========================================
# 2. TIMIT 61 -> 39 音素マッピング
# ==========================================
# TIMITの標準的な61音素を39音素クラスへ縮約するマッピングテーブル (ブランクは0を想定し、音素は1〜61のインデックスとする)
TIMIT_61_TO_39_MAP = {
    'aa': 'aa', 'ao': 'aa', 'ah': 'ah', 'ax': 'ah', 'ax-h': 'ah',
    'er': 'er', 'axr': 'er', 'hh': 'hh', 'hv': 'hh', 'ih': 'ih', 'ix': 'ih',
    'l': 'l', 'el': 'l', 'm': 'm', 'em': 'm', 'n': 'n', 'en': 'n', 'nx': 'n',
    'ng': 'ng', 'eng': 'ng', 'sh': 'sh', 'zh': 'sh',
    'pcl': 'sil', 'tcl': 'sil', 'kcl': 'sil', 'bcl': 'sil', 'dcl': 'sil', 'gcl': 'sil',
    'h#': 'sil', 'pau': 'sil', 'epi': 'sil', 'q': 'sil',
    'uw': 'uw', 'ux': 'uw'
}

PHONEME_LIST = ['<blank>'] + [
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 
    'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 
    'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh'
]
PHONEME_TO_ID = {p: i for i, p in enumerate(PHONEME_LIST)}
ID_TO_PHONEME = {i: p for p, i in PHONEME_TO_ID.items()}

def map_phoneme_to_39(phn_str):
    if phn_str in TIMIT_61_TO_39_MAP:
        return TIMIT_61_TO_39_MAP[phn_str]
    return phn_str


# ==========================================
# 3. RSNNモジュール (Stage 1 用)
# ==========================================
class BaseNeuronGroup:
    def __init__(self, n_neurons, cfg: Config, device=device):
        self.n = n_neurons
        self.cfg = cfg
        self.device = device
        self.alpha = float(np.exp(-cfg.time_step_dt / cfg.tau_mem))
        self.decay_r = float(np.exp(-cfg.time_step_dt / cfg.tau_r))
        self.decay_d = float(np.exp(-cfg.time_step_dt / cfg.tau_d))
        self.c_rd = 1.0 / (cfg.tau_r * cfg.tau_d)
        self.reset_state()

    def reset_state(self):
        self.v = torch.zeros(self.n, device=self.device)
        self.z = torch.zeros(self.n, device=self.device)
        self.z_bar = torch.zeros(self.n, device=self.device)
        self.z_bar_bar = torch.zeros(self.n, device=self.device)
        self.q = torch.full((self.n,), self.c_rd, device=self.device)
        self.psi = torch.zeros(self.n, device=self.device)
        self.b = torch.full((self.n,), self.cfg.v_th, device=self.device)
        self.ref_counts = torch.zeros(self.n, device=self.device)

    def _update_spikes_and_filters(self, v_next_raw, b_next):
        self.psi = (self.cfg.gamma_d / self.cfg.v_th) * torch.clamp(1.0 - torch.abs((v_next_raw - b_next) / self.cfg.v_th), min=0.0)
        is_currently_refractory = self.ref_counts > 0
        self.psi[is_currently_refractory] = 0.0
        spike_cond = (v_next_raw > b_next) & (~is_currently_refractory)
        self.z = spike_cond.float()

        v_next = torch.where(is_currently_refractory, self.v, v_next_raw - self.z * b_next)
        self.ref_counts = torch.clamp(self.ref_counts - self.cfg.time_step_dt, min=0.0)
        self.ref_counts = torch.where(self.z > 0, torch.tensor(self.cfg.t_ref, device=self.device), self.ref_counts)

        # ダブル指数フィルタの更新
        self.z_bar_bar = self.decay_d * self.z_bar_bar + self.q
        self.q = self.decay_r * self.q + self.c_rd * self.z
        self.z_bar = self.alpha * self.z_bar + self.z_bar_bar

        self.v = v_next
        self.b = b_next

class LIFGroup(BaseNeuronGroup):
    def __init__(self, n_neurons, cfg: Config, **kwargs):
        super().__init__(n_neurons, cfg, **kwargs)
        self.beta = 0.0

    def step(self, input_current):
        v_next_raw = self.alpha * self.v + input_current
        b_next = torch.full((self.n,), self.cfg.v_th, device=self.device)
        self._update_spikes_and_filters(v_next_raw, b_next)

class ALIFGroup(BaseNeuronGroup):
    def __init__(self, n_neurons, cfg: Config, **kwargs):
        super().__init__(n_neurons, cfg, **kwargs)
        self.beta = cfg.beta
        self.rho = float(np.exp(-cfg.time_step_dt / cfg.tau_alif))
        self.a = torch.zeros(self.n, device=self.device)

    def reset_state(self):
        super().reset_state()
        self.a = torch.zeros(self.n, device=self.device)

    def step(self, input_current):
        self.a = self.rho * self.a + self.z
        v_next_raw = self.alpha * self.v + input_current
        b_next = self.cfg.v_th + self.beta * self.a
        self._update_spikes_and_filters(v_next_raw, b_next)


class EPropOptimizer:
    def __init__(self, cfg: Config, decay_r: float, decay_d: float, c_rd: float, device=device):
        self.cfg = cfg
        self.device = device
        self.decay_r = decay_r
        self.decay_d = decay_d
        self.c_rd = c_rd
        self.reset_traces()

    def reset_traces(self):
        n = self.cfg.n_neurons
        o = self.cfg.n_outputs
        self.e_trace = torch.zeros((n, n), device=self.device)
        self.eps_b = torch.zeros((n, n), device=self.device)
        self.q = torch.zeros((n, n), device=self.device)
        self.e_bar_bar = torch.zeros((n, n), device=self.device)
        self.grad_w_rec = torch.zeros((n, n), device=self.device)
        self.grad_w_out = torch.zeros((o, n), device=self.device)
        self.grad_w_rec_accumulated = torch.zeros((n, n), device=self.device)
        self.grad_w_out_accumulated = torch.zeros((o, n), device=self.device)
        self.spike_buffer = torch.zeros(n, device=self.device)
        self.e_trace_sum_t_delays = torch.zeros((n, n), device=self.device)
        self.step_counter = 0

    def update_traces_and_gradients(self, neurons, z_bar_curr, z_bar_prev, z_bar_prev_prev, psi_prev, L_t, error_t):
        for idx, g in enumerate(neurons):
            offset = 0 if idx == 0 else neurons[0].n
            n_g, beta, rho = g.n, g.beta, getattr(g, "rho", 0.0)
            psi_curr = g.psi.unsqueeze(1)
            psi_prev_g = psi_prev[offset:offset + n_g].unsqueeze(1)

            if beta > 0:
                self.eps_b[offset:offset + n_g, :] = psi_prev_g * z_bar_prev_prev.unsqueeze(0) + (rho - beta * psi_prev_g) * self.eps_b[offset:offset + n_g, :]
            self.e_trace[offset:offset + n_g, :] = psi_curr * (z_bar_prev.unsqueeze(0) - beta * self.eps_b[offset:offset + n_g, :])

        self.e_bar_bar = self.decay_d * self.e_bar_bar + self.q
        self.q = self.decay_r * self.q + self.e_trace

        self.grad_w_rec_accumulated += L_t.unsqueeze(1) * self.e_bar_bar
        self.grad_w_out_accumulated += torch.outer(error_t, z_bar_curr)

        self.spike_buffer += torch.cat([g.z for g in neurons])
        self.e_trace_sum_t_delays += self.e_trace
        self.step_counter += 1

        if self.step_counter >= self.cfg.t_delay:
            self.grad_w_rec = self.grad_w_rec_accumulated.clone()
            self.grad_w_out = self.grad_w_out_accumulated.clone()
            
            f_bar = self.spike_buffer / self.cfg.t_delay
            reg_signal = (self.cfg.lambda_reg / self.cfg.t_delay) * (f_bar - self.cfg.f_star)
            self.grad_w_rec += reg_signal.unsqueeze(1) * self.e_trace_sum_t_delays

            self.spike_buffer.zero_()
            self.e_trace_sum_t_delays.zero_()
            self.grad_w_rec_accumulated.zero_()
            self.grad_w_out_accumulated.zero_()
            self.step_counter = 0
            return True
        return False

    def apply_weight_update(self, w_rec, w_out):
        w_rec -= self.cfg.eta_rec * self.grad_w_rec + self.cfg.eta_rec * (2.0 * self.cfg.lambda_w) * w_rec
        w_out -= self.cfg.eta_out * self.grad_w_out + self.cfg.eta_out * (2.0 * self.cfg.lambda_w) * w_out
        self.grad_w_rec.zero_()
        self.grad_w_out.zero_()
        return w_rec, w_out


class PredictiveEPropNet:
    def __init__(self, cfg: Config, device=device):
        self.cfg = cfg
        self.device = device
        self.lif = LIFGroup(cfg.n_lif, cfg=cfg, device=device)
        self.alif = ALIFGroup(cfg.n_alif, cfg=cfg, device=device)
        self.neurons = [self.lif, self.alif]

        self.w_in = torch.randn(cfg.n_neurons, cfg.n_inputs, device=device) / np.sqrt(cfg.g)
        self.w_fb = torch.randn(cfg.n_neurons, cfg.n_outputs, device=device) / np.sqrt(cfg.g)
        self.w_rec_mask = (torch.rand(cfg.n_neurons, cfg.n_neurons, device=device) < cfg.density).float()
        self.w_rec_mask.fill_diagonal_(0.0)
        self.w_rec = (torch.randn(cfg.n_neurons, cfg.n_neurons, device=device) / np.sqrt(cfg.n_neurons)) * self.w_rec_mask
        self.w_out = torch.randn(cfg.n_outputs, cfg.n_neurons, device=device) / np.sqrt(cfg.n_neurons)
        self.B = torch.randn(cfg.n_neurons, cfg.n_outputs, device=device) / np.sqrt(cfg.n_neurons)

        self.s = torch.zeros(cfg.n_neurons, device=device)
        self.optimizer = EPropOptimizer(cfg=self.cfg, decay_r=self.lif.decay_r, decay_d=self.lif.decay_d, c_rd=self.lif.c_rd, device=device)
        self.requires_grad = True

    def reset_state(self):
        self.lif.reset_state()
        self.alif.reset_state()
        self.optimizer.reset_traces()
        self.s.zero_()

    @torch.no_grad()
    def run_forward(self, inputs):
        self.reset_state()
        timesteps = inputs.shape[0]
        outputs = torch.zeros((timesteps, self.cfg.n_outputs), device=self.device)
        z_bar_bar_seq = torch.zeros((timesteps, self.cfg.n_neurons), device=self.device)

        z_bar_prev = torch.zeros(self.cfg.n_neurons, device=self.device)
        z_bar_prev_prev = torch.zeros(self.cfg.n_neurons, device=self.device)
        psi_prev = torch.zeros(self.cfg.n_neurons, device=self.device)

        for t in range(timesteps):
            x_t = inputs[t]
            all_z_bar_bar = torch.cat([self.lif.z_bar_bar, self.alif.z_bar_bar])
            all_z_bar = torch.cat([self.lif.z_bar, self.alif.z_bar])

            z_bar_bar_seq[t] = all_z_bar_bar

            y_t = self.w_out @ all_z_bar_bar
            outputs[t] = y_t
            d_t = y_t - x_t  # 予測誤差

            i_rec = self.w_rec @ all_z_bar_bar
            i_in = self.w_in @ d_t
            i_fb = self.w_fb @ y_t
            
            theta = self.cfg.time_step_dt / self.cfg.tau_s
            self.s = self.s + (self.cfg.mu_s - self.s) * theta + self.cfg.sigma_s * theta * torch.randn(self.cfg.n_neurons, device=self.device)

            total_current = i_in + i_rec + i_fb + self.cfg.I_bias # + self.s

            self.lif.step(total_current[:self.cfg.n_lif])
            self.alif.step(total_current[self.cfg.n_lif:])

            if self.requires_grad:
                L_t = self.B @ d_t
                all_psi = torch.cat([self.lif.psi, self.alif.psi])
                should_update = self.optimizer.update_traces_and_gradients(self.neurons, all_z_bar, z_bar_prev, z_bar_prev_prev, psi_prev, L_t, d_t)
                
                if should_update:
                    self.w_rec, self.w_out = self.optimizer.apply_weight_update(self.w_rec, self.w_out)
                    self.w_rec *= self.w_rec_mask

                z_bar_prev_prev = z_bar_prev.clone()
                z_bar_prev = all_z_bar.clone()
                psi_prev = all_psi.clone()

        loss = torch.mean((outputs - inputs) ** 2).item()
        return outputs, z_bar_bar_seq, loss

# ==========================================
# 4. CTCデコーダモジュール (Stage 2 用)
# ==========================================
class CTCDecoder(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.w_dec = nn.Linear(in_features, num_classes)
        
    def forward(self, z_bar_bar_seq):
        logits = self.w_dec(z_bar_bar_seq)
        return F.log_softmax(logits, dim=-1)


# ==========================================
# 5. Utils (データロード, PER評価 & ビームサーチ)
# ==========================================
def load_pickle_dataset(feature_path, target_path):
    """
    TIMIT processing scriptで生成されたpickleファイルを読み込む。
    features: List of (T, N_features) numpy arrays
    targets: List of (T,) numpy arrays or lists containing phoneme IDs
    """
    if not os.path.exists(feature_path) or not os.path.exists(target_path):
        raise FileNotFoundError(f"Pickle files not found: {feature_path} or {target_path}. "
                                f"Please ensure the data has been processed and paths are correct.")
    
    with open(feature_path, 'rb') as f:
        features = pickle.load(f)
    
    with open(target_path, 'rb') as f:
        targets = pickle.load(f)
        
    return features, targets

def process_target_sequence(target_arr):
    """ CTCのターゲットとして連続重複を削除した配列を構築する """
    target_unique = [target_arr[0]]
    for val in target_arr[1:]:
        if val != target_unique[-1]:
            target_unique.append(val)
    return np.array(target_unique, dtype=np.int64)

def ctc_beam_search_decoder(probs_np, beam_width, blank=0):
    T, C = probs_np.shape
    beam = {tuple(): [probs_np[0, blank], 0.0]}
    
    for c in range(1, C):
        beam[(c,)] = [0.0, probs_np[0, c]]
        
    for t in range(1, T):
        next_beam = {}
        for prefix, (p_b, p_nb) in beam.items():
            p_total = p_b + p_nb
            
            if prefix not in next_beam: next_beam[prefix] = [0.0, 0.0]
            next_beam[prefix][0] += p_total * probs_np[t, blank]
            
            for c in range(1, C):
                p_c = probs_np[t, c]
                if len(prefix) > 0 and c == prefix[-1]:
                    next_beam[prefix][1] += p_nb * p_c
                    new_prefix = prefix + (c,)
                    if new_prefix not in next_beam: next_beam[new_prefix] = [0.0, 0.0]
                    next_beam[new_prefix][1] += p_b * p_c
                else:
                    new_prefix = prefix + (c,)
                    if new_prefix not in next_beam: next_beam[new_prefix] = [0.0, 0.0]
                    next_beam[new_prefix][1] += p_total * p_c
                    
        sorted_beam = sorted(next_beam.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)
        beam = {k: v for k, v in sorted_beam[:beam_width]}
        
    best_path = max(beam.items(), key=lambda x: x[1][0] + x[1][1])[0]
    return list(best_path)

def compute_per(ref_phns, hyp_phns):
    ref_39 = [map_phoneme_to_39(ID_TO_PHONEME[i]) for i in ref_phns if i != 0]
    hyp_39 = [map_phoneme_to_39(ID_TO_PHONEME[i]) for i in hyp_phns if i != 0]
    
    m, n = len(ref_39), len(hyp_39)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_39[i - 1] == hyp_39[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n] / max(m, 1)


# ==========================================
# 6. メインパイプライン (事前学習 -> デコーダ学習 -> 評価)
# ==========================================
if __name__ == '__main__':
    cfg = Config()
    
    # --- 1. Pickleファイルからデータをロード ---
    print("\n[Init] Loading acoustic features and targets from pickle files...")
    try:
        features_list, targets_list = load_pickle_dataset(cfg.features_pickle_path, cfg.targets_pickle_path)
        
        # 訓練データとテストデータに分割 (最初のシーケンスを訓練用、2つ目をテスト用として抽出)
        if len(features_list) >= 2:
            train_input_np = np.array(features_list[0], dtype=np.float32)
            train_target_np = np.array(targets_list[0], dtype=np.int64)
            test_input_np = np.array(features_list[1], dtype=np.float32)
            test_target_np = np.array(targets_list[1], dtype=np.int64)
            print("Successfully split data into train and test sequences.")
        else:
            print("Warning: Only 1 sequence found. Using it for both training and testing.")
            train_input_np = test_input_np = np.array(features_list[0], dtype=np.float32)
            train_target_np = test_target_np = np.array(targets_list[0], dtype=np.int64)
            
        train_target_np = process_target_sequence(train_target_np)
        test_target_np = process_target_sequence(test_target_np)

    except FileNotFoundError as e:
        print(e)
        print("Using dummy data instead to demonstrate the pipeline...")
        train_input_np = test_input_np = np.random.randn(500, cfg.n_inputs).astype(np.float32) * 0.5 + 1.0
        train_target_np = test_target_np = np.random.randint(1, cfg.n_phonemes_out, (15,), dtype=np.int64)

    train_seq_len = train_input_np.shape[0]
    test_seq_len = test_input_np.shape[0]
    
    # 入力次元数の動的調整
    cfg.n_inputs = train_input_np.shape[1]
    cfg.n_outputs = train_input_np.shape[1]
    
    train_input_tensor = torch.tensor(train_input_np, device=device)
    train_target_tensor = torch.tensor(train_target_np, device=device)
    test_input_tensor = torch.tensor(test_input_np, device=device)
    test_target_tensor = torch.tensor(test_target_np, device=device)
    
    train_target_lengths = torch.tensor([len(train_target_tensor)], device=device, dtype=torch.long)
    train_input_lengths = torch.tensor([train_seq_len], device=device, dtype=torch.long)


    # --- 2. [実行1] Stage 1 実行: Predictive E-prop RSNN ---
    print(f"\n[Stage 1] Pre-training RSNN with Predictive E-prop on training sequence (len: {train_seq_len})...")
    rsnn = PredictiveEPropNet(cfg, device=device)
    
    for epoch in range(cfg.stage1_epochs):
        _, _, mse_loss = rsnn.run_forward(train_input_tensor)
        print(f"  Stage 1 - Epoch {epoch+1}/{cfg.stage1_epochs} | MSE Loss: {mse_loss:.5f}")


    # --- 3. [実行2] フリーズ & 切替 ---
    print("\n[Transition] Freezing RSNN parameters and attaching CTC Decoder...")
    rsnn.requires_grad = False  # E-prop更新の停止
    
    decoder = CTCDecoder(cfg.n_neurons, cfg.n_phonemes_out).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=0.01)
    ctc_loss_fn = nn.CTCLoss(blank=cfg.blank_id, zero_infinity=True)


    # --- 4. [実行3] Stage 2 実行: CTCデコーダ学習 ---
    print("\n[Stage 2] Fine-tuning Decoder with CTC Loss on training sequence...")
    decoder.train()
    
    for epoch in range(cfg.stage2_epochs):
        with torch.no_grad():
            _, z_bar_bar_seq, _ = rsnn.run_forward(train_input_tensor)
        
        # CTC Decoder へ入力 (Shape: [T, Batch, Features])
        z_bar_bar_seq = z_bar_bar_seq.unsqueeze(1) 
        
        optimizer.zero_grad()
        log_probs = decoder(z_bar_bar_seq) # [T, 1, 62]
        
        loss = ctc_loss_fn(log_probs, train_target_tensor.unsqueeze(0), train_input_lengths, train_target_lengths)
        loss.backward()
        optimizer.step()
        
        print(f"  Stage 2 - Epoch {epoch+1}/{cfg.stage2_epochs} | CTC Loss: {loss.item():.5f}")


    # --- 5. [実行4] 評価実行: ビームサーチデコードとPER算出 ---
    print(f"\n[Stage 3] Evaluation on test sequence (len: {test_seq_len}): Beam Search Decoding & PER Computation...")
    decoder.eval()
    with torch.no_grad():
        # テストデータをRSNNに通して特徴抽出
        _, test_z_bar_bar_seq, _ = rsnn.run_forward(test_input_tensor)
        test_z_bar_bar_seq = test_z_bar_bar_seq.unsqueeze(1)
        log_probs = decoder(test_z_bar_bar_seq)
        probs_np = torch.exp(log_probs.squeeze(1)).cpu().numpy()

    # ビームサーチデコード
    hyp_path = ctc_beam_search_decoder(probs_np, beam_width=cfg.beam_width, blank=cfg.blank_id)
    ref_path = test_target_tensor.cpu().tolist()
    
    # PER（音素エラー率）の計算
    per = compute_per(ref_path, hyp_path)
    
    print("\n=== Evaluation Results (Test Data) ===")
    print(f"  Reference (Original)   : {[ID_TO_PHONEME[i] for i in ref_path]}")
    print(f"  Reference (39-reduced) : {[map_phoneme_to_39(ID_TO_PHONEME[i]) for i in ref_path]}")
    print(f"  Hypothesis (39-reduced): {[map_phoneme_to_39(ID_TO_PHONEME[i]) for i in hyp_path if i != 0]}")
    print(f"  Phoneme Error Rate(PER): {per * 100:.2f}%")
    print("======================================")