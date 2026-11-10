import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import datetime
import os
import pickle
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

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
    eta_rec: float = 0.001
    eta_out: float = 0.001
    lambda_w: float = 5e-6
    lambda_reg: float = 2.0
    t_delay: int = 10
    f_star: float = 0.01

    # ネットワーク構造
    n_inputs: int = 123     # filter_banks.pickle の次元数 (41 * 3 = 123次元)
    n_outputs: int = 123    # 予測ターゲットも同じ次元数
    n_lif: int = 100        
    n_alif: int = 300       # 表現力確保のため ALIF を多めに配置
    density: float = 0.2

    # CTCパラメータ
    n_phonemes_out: int = 40 # 39音素 + 1 blank
    blank_id: int = 0

    # ニューロン・シナプスダイナミクス
    gamma_d: float = 0.3
    v_th: float = 0.6
    I_bias: float = 0.02
    tau_mem: float = 20.0
    tau_alif: float = 2000.0
    beta: float = 0.02
    tau_r: float = 50.0
    tau_d: float = 125.0
    t_ref: float = 5.0
    g: float = 150.0

    # OUノイズパラメータ
    tau_s: float = 250.0
    sigma_s: float = 0.1
    mu_s: float = 0.0

    # 実行設定
    stage1_epochs: int = 60
    stage2_epochs: int = 30
    beam_width: int = 100
    batch_size: int = 8      
    
    # データパス設定
    data_base_dir: str = "/home/keitaro-sunagawa/Master-Research/TIMIT" # 環境に合わせて変更
    features_pickle_name: str = "filter_banks.pickle"
    targets_pickle_name: str = "reduced_phonems.pickle"

    @property
    def n_neurons(self) -> int:
        return self.n_lif + self.n_alif


# ==========================================
# 2. 音素リスト定義
# ==========================================
PHONEME_LIST = [
    '<blank>', 'sil', 'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'er',
    'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy',
    'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z'
]
PHONEME_TO_ID = {p: i for i, p in enumerate(PHONEME_LIST)}
ID_TO_PHONEME = {i: p for p, i in PHONEME_TO_ID.items()}


# ==========================================
# 3. RSNNモジュール (Stage 1 用・バッチ対応済)
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
        
    def reset_state(self, batch_size):
        self.v = torch.zeros(batch_size, self.n, device=self.device)
        self.z = torch.zeros(batch_size, self.n, device=self.device)
        self.z_bar = torch.zeros(batch_size, self.n, device=self.device)
        self.z_bar_bar = torch.zeros(batch_size, self.n, device=self.device)
        self.q = torch.full((batch_size, self.n), self.c_rd, device=self.device)
        self.psi = torch.zeros(batch_size, self.n, device=self.device)
        self.b = torch.full((batch_size, self.n), self.cfg.v_th, device=self.device)
        self.ref_counts = torch.zeros(batch_size, self.n, device=self.device)

    def _update_spikes_and_filters(self, v_next_raw, b_next):
        self.psi = (self.cfg.gamma_d / self.cfg.v_th) * torch.clamp(1.0 - torch.abs((v_next_raw - b_next) / self.cfg.v_th), min=0.0)
        is_currently_refractory = self.ref_counts > 0
        self.psi[is_currently_refractory] = 0.0
        spike_cond = (v_next_raw > b_next) & (~is_currently_refractory)
        self.z = spike_cond.float()

        v_next = torch.where(is_currently_refractory, self.v, v_next_raw - self.z * b_next)
        self.ref_counts = torch.clamp(self.ref_counts - self.cfg.time_step_dt, min=0.0)
        self.ref_counts = torch.where(self.z > 0, torch.tensor(self.cfg.t_ref, device=self.device), self.ref_counts)

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
        b_next = torch.full_like(self.b, self.cfg.v_th, device=self.device)
        self._update_spikes_and_filters(v_next_raw, b_next)

class ALIFGroup(BaseNeuronGroup):
    def __init__(self, n_neurons, cfg: Config, **kwargs):
        super().__init__(n_neurons, cfg, **kwargs)
        self.beta = cfg.beta
        self.rho = float(np.exp(-cfg.time_step_dt / cfg.tau_alif))

    def reset_state(self, batch_size):
        super().reset_state(batch_size)
        self.a = torch.zeros(batch_size, self.n, device=self.device)

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

    def reset_traces(self, batch_size):
        n = self.cfg.n_neurons
        o = self.cfg.n_outputs
        
        self.e_trace = torch.zeros((batch_size, n, n), device=self.device)
        self.eps_b = torch.zeros((batch_size, n, n), device=self.device)
        self.q = torch.zeros((batch_size, n, n), device=self.device)
        self.e_bar_bar = torch.zeros((batch_size, n, n), device=self.device)
        
        # チャンク用の蓄積バッファ
        self.e_trace_sum_t_delays = torch.zeros((batch_size, n, n), device=self.device)
        self.spike_buffer = torch.zeros((batch_size, n), device=self.device)
        self.grad_w_rec_chunk = torch.zeros((n, n), device=self.device)
        self.grad_w_out_chunk = torch.zeros((o, n), device=self.device)
        
        # シーケンス全体用の総勾配蓄積バッファ（ここでシーケンス最後まで溜め込む）
        self.total_grad_w_rec = torch.zeros((n, n), device=self.device)
        self.total_grad_w_out = torch.zeros((o, n), device=self.device)
        
        self.step_counter = 0

    def update_traces_and_gradients(self, neurons, z_bar_curr, z_bar_prev, z_bar_prev_prev, psi_prev, L_t, error_t, mask_t):
        for idx, g in enumerate(neurons):
            offset = 0 if idx == 0 else neurons[0].n
            n_g, beta, rho = g.n, g.beta, getattr(g, "rho", 0.0)
            
            psi_curr = g.psi.unsqueeze(2) 
            psi_prev_g = psi_prev[:, offset:offset + n_g].unsqueeze(2)

            if beta > 0:
                self.eps_b[:, offset:offset + n_g, :] = psi_prev_g * z_bar_prev_prev.unsqueeze(1) + (rho - beta * psi_prev_g) * self.eps_b[:, offset:offset + n_g, :]
            self.e_trace[:, offset:offset + n_g, :] = psi_curr * (z_bar_prev.unsqueeze(1) - beta * self.eps_b[:, offset:offset + n_g, :])

        self.e_bar_bar = self.decay_d * self.e_bar_bar + self.q
        self.q = self.decay_r * self.q + self.e_trace

        # チャンク用の勾配蓄積
        grad_rec_step = (L_t.unsqueeze(2) * self.e_bar_bar) * mask_t.unsqueeze(2)
        self.grad_w_rec_chunk += grad_rec_step.mean(dim=0)
        
        grad_out_step = torch.bmm(error_t.unsqueeze(2), z_bar_curr.unsqueeze(1)) * mask_t.unsqueeze(2)
        self.grad_w_out_chunk += grad_out_step.mean(dim=0)

        self.spike_buffer += torch.cat([g.z for g in neurons], dim=1) * mask_t
        self.e_trace_sum_t_delays += self.e_trace * mask_t.unsqueeze(2)
        self.step_counter += 1

        # t_delayステップごとに正則化項を計算し、総勾配に合流させる
        if self.step_counter >= self.cfg.t_delay:
            f_bar = self.spike_buffer.mean(dim=0) / self.cfg.t_delay
            reg_signal = (self.cfg.lambda_reg / self.cfg.t_delay) * (f_bar - self.cfg.f_star)
            
            self.grad_w_rec_chunk += reg_signal.unsqueeze(1) * self.e_trace_sum_t_delays.mean(dim=0)
            
            # 総勾配へ加算
            self.total_grad_w_rec += self.grad_w_rec_chunk
            self.total_grad_w_out += self.grad_w_out_chunk
            
            # チャンクバッファをリセット
            self.spike_buffer.zero_()
            self.e_trace_sum_t_delays.zero_()
            self.grad_w_rec_chunk.zero_()
            self.grad_w_out_chunk.zero_()
            self.step_counter = 0

    def finalize_sequence(self):
        # シーケンスの終端で端数（t_delayに満たないステップ）の勾配を合流させる
        if self.step_counter > 0:
            f_bar = self.spike_buffer.mean(dim=0) / self.step_counter
            reg_signal = (self.cfg.lambda_reg / self.cfg.t_delay) * (f_bar - self.cfg.f_star)
            self.grad_w_rec_chunk += reg_signal.unsqueeze(1) * self.e_trace_sum_t_delays.mean(dim=0)
            self.total_grad_w_rec += self.grad_w_rec_chunk
            self.total_grad_w_out += self.grad_w_out_chunk


class PredictiveEPropNet(nn.Module):
    def __init__(self, cfg: Config, device=device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.lif = LIFGroup(cfg.n_lif, cfg=cfg, device=device)
        self.alif = ALIFGroup(cfg.n_alif, cfg=cfg, device=device)
        self.neurons = [self.lif, self.alif]

        self.w_in = torch.randn(cfg.n_neurons, cfg.n_inputs, device=device) / np.sqrt(cfg.g)
        self.w_fb = torch.randn(cfg.n_neurons, cfg.n_outputs, device=device) / np.sqrt(cfg.g)
        
        self.w_rec_mask = (torch.rand(cfg.n_neurons, cfg.n_neurons, device=device) < cfg.density).float()
        self.w_rec_mask.fill_diagonal_(0.0)
        
        self.w_rec = nn.Parameter((torch.randn(cfg.n_neurons, cfg.n_neurons, device=device) / np.sqrt(cfg.n_neurons)) * self.w_rec_mask)
        self.w_out = nn.Parameter(torch.randn(cfg.n_outputs, cfg.n_neurons, device=device) / np.sqrt(cfg.n_neurons))
        
        self.B = torch.randn(cfg.n_neurons, cfg.n_outputs, device=device) / np.sqrt(cfg.n_neurons)

        self.eprop_tracker = EPropOptimizer(cfg=self.cfg, decay_r=self.lif.decay_r, decay_d=self.lif.decay_d, c_rd=self.lif.c_rd, device=device)
        self.requires_grad = True

    def reset_state(self, batch_size):
        self.s = torch.zeros(batch_size, self.cfg.n_neurons, device=self.device)
        self.lif.reset_state(batch_size)
        self.alif.reset_state(batch_size)
        self.eprop_tracker.reset_traces(batch_size)

    @torch.no_grad()
    def run_forward(self, inputs, mask, adam_optimizer=None):
        timesteps, batch_size, _ = inputs.shape
        self.reset_state(batch_size)
        
        outputs = torch.zeros((timesteps, batch_size, self.cfg.n_outputs), device=self.device)
        z_bar_bar_seq = torch.zeros((timesteps, batch_size, self.cfg.n_neurons), device=self.device)

        z_bar_prev = torch.zeros(batch_size, self.cfg.n_neurons, device=self.device)
        z_bar_prev_prev = torch.zeros(batch_size, self.cfg.n_neurons, device=self.device)
        psi_prev = torch.zeros(batch_size, self.cfg.n_neurons, device=self.device)

        for t in range(timesteps):
            x_t = inputs[t]             
            m_t = mask[t].unsqueeze(1)  

            all_z_bar_bar = torch.cat([self.lif.z_bar_bar, self.alif.z_bar_bar], dim=1)
            all_z_bar = torch.cat([self.lif.z_bar, self.alif.z_bar], dim=1)

            z_bar_bar_seq[t] = all_z_bar_bar

            y_t = all_z_bar_bar @ self.w_out.T
            outputs[t] = y_t
            d_t = (y_t - x_t) * m_t  

            i_rec = all_z_bar_bar @ self.w_rec.T
            i_in = d_t @ self.w_in.T
            i_fb = y_t @ self.w_fb.T
            
            # OUノイズを加えてネットワークの探索（発火）を促進
            theta = self.cfg.time_step_dt / self.cfg.tau_s
            self.s = self.s + (self.cfg.mu_s - self.s) * theta + self.cfg.sigma_s * theta * torch.randn(batch_size, self.cfg.n_neurons, device=self.device)
            total_current = i_in + i_rec + i_fb + self.cfg.I_bias # + self.s

            self.lif.step(total_current[:, :self.cfg.n_lif])
            self.alif.step(total_current[:, self.cfg.n_lif:])
            
            self.lif.z *= m_t
            self.alif.z *= m_t

            if self.requires_grad:
                L_t = d_t @ self.B.T
                all_psi = torch.cat([self.lif.psi, self.alif.psi], dim=1)
                
                self.eprop_tracker.update_traces_and_gradients(
                    self.neurons, all_z_bar, z_bar_prev, z_bar_prev_prev, psi_prev, L_t, d_t, m_t
                )

                z_bar_prev_prev = z_bar_prev.clone()
                z_bar_prev = all_z_bar.clone()
                psi_prev = all_psi.clone()

        # --- シーケンス終了後に1回だけAdamを更新する ---
        if self.requires_grad and (adam_optimizer is not None):
            self.eprop_tracker.finalize_sequence()
            
            self.w_rec.grad = self.eprop_tracker.total_grad_w_rec.clone()
            self.w_out.grad = self.eprop_tracker.total_grad_w_out.clone()
            
            # Weight Decay (L2正則化)
            self.w_rec.grad += (2.0 * self.cfg.lambda_w) * self.w_rec.data
            self.w_out.grad += (2.0 * self.cfg.lambda_w) * self.w_out.data
            
            adam_optimizer.step()
            adam_optimizer.zero_grad()
            
            # 自己結合なし・疎結合のマスクを適用して構造を維持
            self.w_rec.data *= self.w_rec_mask

        mse_loss = torch.sum(((outputs - inputs) ** 2) * mask.unsqueeze(2)) / (mask.sum() * self.cfg.n_outputs)
        return outputs, z_bar_bar_seq, mse_loss.item()


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
# 5. Utils & DataLoader (バッチ対応)
# ==========================================
def load_pickle_dataset(feature_path, target_path):
    with open(feature_path, 'rb') as f:
        features = pickle.load(f)
    with open(target_path, 'rb') as f:
        targets = pickle.load(f)
    return features, targets

def process_target_sequence(target_arr):
    target_unique = [target_arr[0]]
    for val in target_arr[1:]:
        if val != target_unique[-1]:
            target_unique.append(val)
    return np.array(target_unique, dtype=np.int64)

class TimitDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = torch.tensor(self.features[idx], dtype=torch.float32)
        tgt = np.array(self.targets[idx], dtype=np.int64) + 1
        tgt = process_target_sequence(tgt)
        label = torch.tensor(tgt, dtype=torch.long)
        return feat, label

def collate_fn(batch):
    features, labels = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)
    target_lengths = torch.tensor([l.size(0) for l in labels], dtype=torch.long)
    
    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=False, padding_value=0.0)
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    
    max_len = padded_features.size(0)
    mask = torch.arange(max_len).unsqueeze(1) < lengths.unsqueeze(0)
    
    return padded_features.to(device), padded_labels.to(device), lengths.to(device), target_lengths.to(device), mask.float().to(device)


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
    ref_str = [ID_TO_PHONEME[i] for i in ref_phns if i != 0]
    hyp_str = [ID_TO_PHONEME[i] for i in hyp_phns if i != 0]
    
    m, n = len(ref_str), len(hyp_str)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_str[i - 1] == hyp_str[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n] / max(m, 1)


# ==========================================
# 6. メインパイプライン
# ==========================================
if __name__ == '__main__':
    cfg = Config()
    
    print(f"\n[Init] Loading acoustic features and targets from '{cfg.data_base_dir}'...")
    
    train_features_path = os.path.join(cfg.data_base_dir, "train_pickle", cfg.features_pickle_name)
    train_targets_path = os.path.join(cfg.data_base_dir, "train_pickle", cfg.targets_pickle_name)
    test_features_path = os.path.join(cfg.data_base_dir, "test_pickle", cfg.features_pickle_name)
    test_targets_path = os.path.join(cfg.data_base_dir, "test_pickle", cfg.targets_pickle_name)

    train_feat_list, train_tgt_list = load_pickle_dataset(train_features_path, train_targets_path)
    test_feat_list, test_tgt_list = load_pickle_dataset(test_features_path, test_targets_path)

    cfg.n_inputs = train_feat_list[0].shape[1]
    cfg.n_outputs = train_feat_list[0].shape[1]

    train_dataset = TimitDataset(train_feat_list, train_tgt_list)
    test_dataset = TimitDataset(test_feat_list, test_tgt_list)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


    # --- 2. [実行1] Stage 1 実行: Predictive E-prop RSNN ---
    print(f"\n[Stage 1] Pre-training RSNN with Predictive E-prop (Batch Size: {cfg.batch_size})...")
    rsnn = PredictiveEPropNet(cfg, device=device)
    
    stage1_optimizer = optim.Adam(rsnn.parameters(), lr=0.002) # 全体バッチ更新向けに学習率を調整
    
    for epoch in range(cfg.stage1_epochs):
        epoch_mse_loss = 0.0
        for batch_idx, (inputs, _, _, _, mask) in enumerate(train_loader):
            _, _, mse_loss = rsnn.run_forward(inputs, mask, adam_optimizer=stage1_optimizer)
            epoch_mse_loss += mse_loss
            
        avg_mse_loss = epoch_mse_loss / len(train_loader)
        print(f"  Stage 1 - Epoch {epoch+1}/{cfg.stage1_epochs} | Avg MSE Loss: {avg_mse_loss:.5f}")


    # --- 3. [実行2] フリーズ & 切替 ---
    print("\n[Transition] Freezing RSNN parameters and attaching CTC Decoder...")
    rsnn.requires_grad = False
    decoder = CTCDecoder(cfg.n_neurons, cfg.n_phonemes_out).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    ctc_loss_fn = nn.CTCLoss(blank=cfg.blank_id, zero_infinity=True)


    # --- 4. [実行3] Stage 2 実行: CTCデコーダ学習 ---
    print(f"\n[Stage 2] Fine-tuning Decoder with CTC Loss (Batch Size: {cfg.batch_size})...")
    decoder.train()
    
    for epoch in range(cfg.stage2_epochs):
        epoch_ctc_loss = 0.0
        for batch_idx, (inputs, targets, input_lengths, target_lengths, mask) in enumerate(train_loader):
            
            with torch.no_grad():
                _, z_bar_bar_seq, _ = rsnn.run_forward(inputs, mask)
            
            optimizer.zero_grad()
            log_probs = decoder(z_bar_bar_seq)
            
            loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            
            epoch_ctc_loss += loss.item()
            
        avg_ctc_loss = epoch_ctc_loss / len(train_loader)
        print(f"  Stage 2 - Epoch {epoch+1}/{cfg.stage2_epochs} | Avg CTC Loss: {avg_ctc_loss:.5f}")


    # --- 5. [実行4] 評価実行: ビームサーチデコードとPER算出 ---
    print(f"\n[Stage 3] Evaluation on test sequences...")
    decoder.eval()
    
    total_per = 0.0
    for i, (inputs, targets, input_lengths, _, mask) in enumerate(test_loader):
        with torch.no_grad():
            _, test_z_bar_bar_seq, _ = rsnn.run_forward(inputs, mask)
            log_probs = decoder(test_z_bar_bar_seq)
            probs_np = torch.exp(log_probs[:, 0, :]).cpu().numpy()

        hyp_path = ctc_beam_search_decoder(probs_np, beam_width=cfg.beam_width, blank=cfg.blank_id)
        ref_path = targets[0].cpu().tolist()
        
        per = compute_per(ref_path, hyp_path)
        total_per += per
        
        if i < 3:
            print(f"\n  [Test Sample {i+1}]")
            print(f"    Reference              : {[ID_TO_PHONEME[idx] for idx in ref_path if idx != 0]}")
            print(f"    Hypothesis             : {[ID_TO_PHONEME[idx] for idx in hyp_path if idx != 0]}")
            print(f"    PER                    : {per * 100:.2f}%")

    avg_per = total_per / len(test_loader)
    print("\n======================================")
    print(f"  Final Average Phoneme Error Rate (PER) on Test Set: {avg_per * 100:.2f}%")
    print("======================================")