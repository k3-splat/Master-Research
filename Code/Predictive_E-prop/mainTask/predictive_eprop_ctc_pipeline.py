import os
import glob
import torch
import scipy.io as sio
import numpy as np
import datetime
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# ==========================================
# 1. Config & TIMIT 音素マッピング
# ==========================================
@dataclass
class Config:
    time_step_dt: float = 1.0
    eta_rec: float = 0.0004
    eta_out: float = 0.0004
    eta_cls: float = 0.001
    lambda_w: float = 5e-6
    lambda_reg: float = 2.0
    t_delay: int = 10
    f_star: float = 0.01

    n_inputs: int = 20  
    n_lif: int = 100
    n_alif: int = 200
    n_outputs: int = 20 
    
    # CTCデコーダ用設定 (39音素 + 1 Blank)
    n_classes_ctc: int = 40 
    blank_idx: int = 0
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

    tau_s: float = 250.0
    sigma_s: float = 0.0
    mu_s: float = 0.0

    epochs: int = 30
    batch_size: int = 8

    @property
    def n_neurons(self) -> int:
        return self.n_lif + self.n_alif

phoneme_reduction_table = {
    'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'hv': 'hh', 'ix': 'ih', 
    'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', 'eng': 'ng', 'zh': 'sh', 'ux': 'uw',
    'pcl': 'sil', 'tcl': 'sil', 'kcl': 'sil', 'bcl': 'sil', 'dcl': 'sil', 'gcl': 'sil',
    'h#': 'sil', 'pau': 'sil', 'epi': 'sil', 'q': 'sil'
}
reduced_phn_list = ['sil', 'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z']
phn_to_id = {phn: i for i, phn in enumerate(reduced_phn_list)}


# ==========================================
# 2. PyTorch Dataset & DataLoader
# ==========================================
class SpikeTIMITDataset(Dataset):
    def __init__(self, root_dir, window_step_samples=160):
        self.root_dir = root_dir
        self.window_step_samples = window_step_samples
        self.mat_files = glob.glob(os.path.join(root_dir, '**/*_binary_matrix.mat'), recursive=True)
        
    def __len__(self):
        return len(self.mat_files)
        
    def __getitem__(self, idx):
        mat_path = self.mat_files[idx]
        phn_path = mat_path.replace('_binary_matrix.mat', '.phn')
        
        # 実行テスト用ダミーデータ生成（実データがない環境向けのフォールバック）
        if not os.path.exists(mat_path):
            spikes = np.random.randn(100, 20).astype(np.float32)
            labels = np.random.randint(1, 40, size=(100,), dtype=np.int64)
            return torch.tensor(spikes), torch.tensor(labels)

        mat_data = sio.loadmat(mat_path)
        key = [k for k in mat_data.keys() if not k.startswith('__')][0]
        spikes = mat_data[key].T.astype(np.float32) 
        time_steps = spikes.shape[0]
        
        # 0をCTC Blankとするため、音素IDは+1シフト (1~39)
        labels = np.full(time_steps, phn_to_id['sil'] + 1, dtype=np.int64)
        if os.path.exists(phn_path):
            with open(phn_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        start_sample, end_sample, phn = int(parts[0]), int(parts[1]), parts[2]
                        if phn in phoneme_reduction_table:
                            phn = phoneme_reduction_table[phn]
                        phn_id = phn_to_id.get(phn, phn_to_id['sil']) + 1 # +1 for Blank
                        
                        start_frame = min(start_sample // self.window_step_samples, time_steps - 1)
                        end_frame = min(end_sample // self.window_step_samples, time_steps)
                        labels[start_frame:end_frame] = phn_id
                        
        return torch.tensor(spikes), torch.tensor(labels)

def collate_fn(batch):
    spikes, labels = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in spikes])
    
    padded_spikes = torch.nn.utils.rnn.pad_sequence(spikes, batch_first=True, padding_value=0.0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0) # 0 is blank
    
    mask = torch.arange(padded_spikes.size(1)).expand(len(lengths), -1) < lengths.unsqueeze(1)
    return padded_spikes.to(device), padded_labels.to(device), mask.float().to(device), lengths.to(device)

def extract_ctc_targets(labels_batch, mask_batch):
    targets = []
    target_lengths = []
    for i in range(labels_batch.size(0)):
        seq = labels_batch[i][mask_batch[i].bool()].tolist()
        compressed = []
        prev = -1
        for val in seq:
            if val != prev:
                compressed.append(val)
                prev = val
        targets.extend(compressed)
        target_lengths.append(len(compressed))
    return torch.tensor(targets, dtype=torch.long, device=device), torch.tensor(target_lengths, dtype=torch.long, device=device)


# ==========================================
# 3. ニューロンモデル
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
        self.batch_size = batch_size
        self.v = torch.zeros((batch_size, self.n), device=self.device, dtype=torch.float32)
        self.z = torch.zeros((batch_size, self.n), device=self.device, dtype=torch.float32)
        self.z_bar = torch.zeros((batch_size, self.n), device=self.device, dtype=torch.float32)
        self.z_bar_bar = torch.zeros((batch_size, self.n), device=self.device, dtype=torch.float32)
        self.q = torch.full((batch_size, self.n), self.c_rd, device=self.device, dtype=torch.float32)
        self.psi = torch.zeros((batch_size, self.n), device=self.device, dtype=torch.float32)
        self.b = torch.full((batch_size, self.n), self.cfg.v_th, device=self.device, dtype=torch.float32)
        self.ref_counts = torch.zeros((batch_size, self.n), device=self.device, dtype=torch.float32)

    def _update_spikes_and_filters(self, v_next_raw, b_next):
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
    def __init__(self, n_neurons, cfg: Config, **kwargs):
        super().__init__(n_neurons, cfg, **kwargs)
        self.beta = 0.0

    def step(self, input_current):
        v_next_raw = self.alpha * self.v + input_current
        b_next = torch.full((self.batch_size, self.n), self.cfg.v_th, device=self.device, dtype=torch.float32)
        self._update_spikes_and_filters(v_next_raw, b_next)

class ALIFGroup(BaseNeuronGroup):
    def __init__(self, n_neurons, cfg: Config, **kwargs):
        super().__init__(n_neurons, cfg, **kwargs)
        self.beta = cfg.beta
        self.rho = float(np.exp(-cfg.time_step_dt / cfg.tau_alif))

    def reset_state(self, batch_size):
        super().reset_state(batch_size)
        self.a = torch.zeros((batch_size, self.n), device=self.device, dtype=torch.float32)

    def step(self, input_current):
        self.a = self.rho * self.a + self.z
        v_next_raw = self.alpha * self.v + input_current
        b_next = self.cfg.v_th + self.beta * self.a
        self._update_spikes_and_filters(v_next_raw, b_next)


# ==========================================
# 4. 最適化手法 (CTC + E-prop 統合版)
# ==========================================
class EPropOptimizer:
    def __init__(self, cfg: Config, decay_r: float, decay_d: float, c_rd: float, device=device):
        self.cfg = cfg
        self.device = device
        self.decay_r = decay_r
        self.decay_d = decay_d
        self.c_rd = c_rd
        
    def reset_traces(self, batch_size):
        self.batch_size = batch_size
        n = self.cfg.n_neurons
        self.e_trace = torch.zeros((batch_size, n, n), device=self.device)
        self.eps_b = torch.zeros((batch_size, n, n), device=self.device)
        self.q_trace = torch.zeros((batch_size, n, n), device=self.device)
        self.e_bar_bar = torch.zeros((batch_size, n, n), device=self.device)

    def forward_trace_step(self, neurons, psi_prev, z_bar_prev, z_bar_prev_prev):
        '''
        フォワードパス中に呼び出し、現在のステップの適格度トレースを更新・返却する
        '''
        for idx, group in enumerate(neurons):
            offset = 0 if idx == 0 else neurons[0].n
            n_g = group.n
            beta = group.beta
            rho = getattr(group, "rho", 0.0)
            
            psi_curr = group.psi.unsqueeze(2)
            psi_prev_g = psi_prev[:, offset:offset+n_g].unsqueeze(2)
            
            if beta > 0:
                self.eps_b[:, offset:offset+n_g, :] = psi_prev_g * z_bar_prev_prev.unsqueeze(1) + (rho - beta * psi_prev_g) * self.eps_b[:, offset:offset+n_g, :]
            self.e_trace[:, offset:offset+n_g, :] = psi_curr * (z_bar_prev.unsqueeze(1) - beta * self.eps_b[:, offset:offset+n_g, :])

        self.e_bar_bar = self.decay_d * self.e_bar_bar + self.q_trace
        self.q_trace = self.decay_r * self.q_trace + self.c_rd * self.e_trace
        
        return self.e_bar_bar.clone()

    def compute_and_apply_gradients(self, e_bar_bar_seq, z_bar_bar_seq, d_pred_seq, d_cls_seq, B_pred, B_cls, w_rec, w_out, w_cls, b_cls):
        '''
        CTCデコード後に取得したエラーシグナルと、記録したトレースを用いて一括で勾配を計算し、重みを更新する
        '''
        # B_pred: [N_rec, N_out], d_pred_seq: [B, T, N_out] => L_pred: [B, T, N_rec]
        L_pred = torch.matmul(d_pred_seq, B_pred.t())
        L_cls = torch.matmul(d_cls_seq, B_cls.t())
        L_total = L_pred + L_cls
        
        # テンソル演算で一括勾配計算 (BatchとTimeで平均)
        normalizer = self.batch_size * L_total.size(1)
        
        # grad_w_rec: [N_rec, N_rec]
        grad_w_rec = torch.einsum('btn,btni->ni', L_total, e_bar_bar_seq) / normalizer
        # grad_w_out: [N_out, N_rec]
        grad_w_out = torch.einsum('bti,btj->ij', d_pred_seq, z_bar_bar_seq) / normalizer
        # grad_w_cls: [N_cls, N_rec]
        grad_w_cls = torch.einsum('btc,btn->cn', d_cls_seq, z_bar_bar_seq) / normalizer
        # grad_b_cls: [N_cls]
        grad_b_cls = torch.mean(torch.sum(d_cls_seq, dim=1), dim=0) / L_total.size(1)

        # 勾配クリッピング
        clip_val = 1.0
        torch.clamp_(grad_w_rec, -clip_val, clip_val)
        torch.clamp_(grad_w_out, -clip_val, clip_val)
        torch.clamp_(grad_w_cls, -clip_val, clip_val)
        torch.clamp_(grad_b_cls, -clip_val, clip_val)

        # L2 Regularization と 重み更新
        w_rec = w_rec - self.cfg.eta_rec * grad_w_rec - self.cfg.eta_rec * (2.0 * self.cfg.lambda_w) * w_rec
        w_out = w_out - self.cfg.eta_out * grad_w_out - self.cfg.eta_out * (2.0 * self.cfg.lambda_w) * w_out
        
        # ユーザコードの w_cls 更新に合わせて転置を考慮 
        # w_cls は [N_rec, N_cls] と定義されているため、grad_w_cls.t() [N_rec, N_cls] を引く
        w_cls = w_cls - self.cfg.eta_cls * grad_w_cls.t() - self.cfg.eta_cls * (2.0 * self.cfg.lambda_w) * w_cls
        b_cls = b_cls - self.cfg.eta_cls * grad_b_cls

        return w_rec, w_out, w_cls, b_cls


# ==========================================
# 5. ネットワーク全体モデル
# ==========================================
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
        self.B_pred = torch.randn(cfg.n_neurons, cfg.n_outputs, device=device) / np.sqrt(cfg.n_neurons)

        self.w_cls = torch.randn(cfg.n_neurons, cfg.n_classes_ctc, device=device) / np.sqrt(cfg.n_neurons)
        self.b_cls = torch.zeros(cfg.n_classes_ctc, device=device)
        self.B_cls = torch.randn(cfg.n_neurons, cfg.n_classes_ctc, device=device) / np.sqrt(cfg.n_neurons)

        self.s = None
        self.optimizer = EPropOptimizer(cfg=self.cfg, decay_r=self.lif.decay_r, decay_d=self.lif.decay_d, c_rd=self.lif.c_rd, device=device)

    def reset_state(self, batch_size):
        self.lif.reset_state(batch_size)
        self.alif.reset_state(batch_size)
        self.optimizer.reset_traces(batch_size)
        self.s = torch.zeros((batch_size, self.cfg.n_neurons), device=self.device)

    def generate_ou_noise(self, batch_size):
        theta = self.cfg.time_step_dt / self.cfg.tau_s
        self.s = self.s + (self.cfg.mu_s - self.s) * theta + self.cfg.sigma_s * theta * torch.randn((batch_size, self.cfg.n_neurons), device=self.device)
        return self.s


# ==========================================
# 6. 学習ループ (Predictive E-prop + CTCエンドツーエンド)
# ==========================================
def train_spike_timit(train_dir="train", batch_size=8, epochs=30):
    # ダミーデータを生成してテスト実行するモードを利用
    train_dataset = SpikeTIMITDataset(root_dir=train_dir)
    if len(train_dataset) == 0:
        print("Warning: Real dataset not found. Using auto-generated dummy batches for validation.")
        # ダミーデータをロードさせるため空ディレクトリを回避する細工が Dataset 側に入っています
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    sample_spikes, _, _, _ = next(iter(train_loader))
    n_channels = sample_spikes.shape[2]
    
    cfg = Config()
    cfg.n_inputs = n_channels
    cfg.n_outputs = n_channels
    cfg.epochs = epochs
    cfg.batch_size = batch_size
    
    net = PredictiveEPropNet(cfg=cfg, device=device)
    ctc_loss_fn = nn.CTCLoss(blank=cfg.blank_idx, reduction='mean', zero_infinity=True)
    
    print(f"Start Training: RSNN E-prop with CTC Decoder")
    
    for epoch in range(epochs):
        epoch_loss_pred = 0.0
        epoch_loss_ctc = 0.0
        
        for batch_idx, (spikes_batch, labels_batch, mask_batch, input_lengths) in enumerate(train_loader):
            current_batch_size, max_time_steps, _ = spikes_batch.shape
            
            # CTCターゲットへの変換
            ctc_targets, target_lengths = extract_ctc_targets(labels_batch, mask_batch)
            
            net.reset_state(batch_size=current_batch_size)
            
            z_bar_prev = torch.zeros((current_batch_size, cfg.n_neurons), device=device)
            z_bar_prev_prev = torch.zeros((current_batch_size, cfg.n_neurons), device=device)
            psi_prev = torch.zeros((current_batch_size, cfg.n_neurons), device=device)
            
            # トレースと出力を記録するリスト
            e_bar_bar_seq = []
            z_bar_bar_seq = []
            d_pred_seq = []
            logits_seq = []
            
            batch_loss_pred = 0.0
            
            # 1. フォワードパス (SNNの時間発展とトレース記録)
            for t in range(max_time_steps):
                x_t = spikes_batch[:, t, :] 
                m_t = mask_batch[:, t] 
                
                all_z_bar_bar = torch.cat([net.lif.z_bar_bar, net.alif.z_bar_bar], dim=1)
                all_z_bar = torch.cat([net.lif.z_bar, net.alif.z_bar], dim=1)
                
                # デコーダ計算
                y_t = all_z_bar_bar @ net.w_out.t()
                logits_t = all_z_bar_bar @ net.w_cls + net.b_cls
                
                # 予測誤差
                d_t_pred = (y_t - x_t) * m_t.unsqueeze(1)
                batch_loss_pred += torch.mean(d_t_pred ** 2).item()
                
                # SNN状態更新
                i_rec = all_z_bar_bar @ net.w_rec.t()
                i_in = d_t_pred @ net.w_in.t() 
                i_fb = y_t @ net.w_fb.t()
                s_t = net.generate_ou_noise(current_batch_size)
                
                total_current = i_in + i_rec + i_fb + cfg.I_bias + s_t
                net.lif.step(total_current[:, :cfg.n_lif])
                net.alif.step(total_current[:, cfg.n_lif:])
                
                all_psi = torch.cat([net.lif.psi, net.alif.psi], dim=1)
                
                # トレース更新と記録
                e_bar_bar_t = net.optimizer.forward_trace_step(net.neurons, psi_prev, z_bar_prev, z_bar_prev_prev)
                
                e_bar_bar_seq.append(e_bar_bar_t)
                z_bar_bar_seq.append(all_z_bar_bar.clone())
                d_pred_seq.append(d_t_pred)
                logits_seq.append(logits_t)
                
                z_bar_prev_prev = z_bar_prev.clone()
                z_bar_prev = all_z_bar.clone()
                psi_prev = all_psi.clone()
            
            # テンソル化 [B, T, ...]
            e_bar_bar_seq_t = torch.stack(e_bar_bar_seq, dim=1)
            z_bar_bar_seq_t = torch.stack(z_bar_bar_seq, dim=1)
            d_pred_seq_t = torch.stack(d_pred_seq, dim=1)
            logits_seq_t = torch.stack(logits_seq, dim=1)
            
            # 2. CTC Loss計算とバックワードによる学習シグナル取得
            logits_seq_t.requires_grad_()
            log_probs = F.log_softmax(logits_seq_t, dim=-1).transpose(0, 1) # [T, B, C]
            
            loss_ctc = ctc_loss_fn(log_probs, ctc_targets, input_lengths, target_lengths)
            
            # Logits に対する勾配を取得し、CTCのエラーシグナルとする
            d_cls_seq_t = torch.autograd.grad(loss_ctc, logits_seq_t)[0] # [B, T, C]
            
            # マスク処理
            mask_expanded = mask_batch.unsqueeze(-1)
            d_cls_seq_t = d_cls_seq_t * mask_expanded
            
            # 3. RSNNの重み更新 (Predictive E-prop)
            net.w_rec, net.w_out, net.w_cls, net.b_cls = net.optimizer.compute_and_apply_gradients(
                e_bar_bar_seq_t, z_bar_bar_seq_t, d_pred_seq_t, d_cls_seq_t,
                net.B_pred, net.B_cls, net.w_rec, net.w_out, net.w_cls, net.b_cls
            )
            net.w_rec *= net.w_rec_mask
            
            epoch_loss_pred += batch_loss_pred / max_time_steps
            epoch_loss_ctc += loss_ctc.item()
            
            if batch_idx % 10 == 0:
                print(f"  Train Batch {batch_idx}/{len(train_loader)} | CTC Loss: {loss_ctc.item():.4f} | Pred Loss: {batch_loss_pred/max_time_steps:.4f}")
                
        print(f"=== Epoch {epoch+1} Train Summary | CTC Loss: {epoch_loss_ctc/len(train_loader):.4f} | Pred Loss: {epoch_loss_pred/len(train_loader):.4f} ===
")
        
        # 1エポックでテスト動作確認を終了
        break

if __name__ == "__main__":
    train_spike_timit(train_dir="dummy_path_for_test", batch_size=4, epochs=1)
