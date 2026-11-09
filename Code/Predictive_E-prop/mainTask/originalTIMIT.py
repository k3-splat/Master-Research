import os
import glob
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# ==========================================
# 1. Config & 音素一覧
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

    n_inputs: int = 39   # MFCC (39次元) の場合
    n_lif: int = 100
    n_alif: int = 200
    n_outputs: int = 39 
    n_classes: int = 39  # 縮約後の音素クラス数
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

# 39音素リスト（silence ID = 0）
reduced_phn_list = [
    'sil', 'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'er',
    'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy',
    'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z'
]
SIL_ID = 0

# ==========================================
# 2. Pickle 専用 PyTorch Dataset & DataLoader
# ==========================================
class TimitPickleDataset(Dataset):
    def __init__(self, target_dir: str, feature_type: str = 'mfcc'):
        """
        target_dir: '.../datasets/timit_processed/train' などのディレクトリ
        feature_type: 'mfcc' または 'fbank'
        """
        feat_name = 'mfccs.pickle' if feature_type == 'mfcc' else 'filter_banks.pickle'
        feat_path = os.path.join(target_dir, feat_name)
        phn_path = os.path.join(target_dir, 'reduced_phonems.pickle')

        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Feature file not found: {feat_path}")
        if not os.path.exists(phn_path):
            raise FileNotFoundError(f"Label file not found: {phn_path}")

        with open(feat_path, 'rb') as f:
            self.features = pickle.load(f)
        with open(phn_path, 'rb') as f:
            self.labels = pickle.load(f)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return feat, label

def collate_fn(batch):
    features, labels = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)

    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=SIL_ID)

    mask = torch.arange(padded_features.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return padded_features.to(device), padded_labels.to(device), mask.float().to(device)


# ==========================================
# 3. ニューロンモデル (バッチ対応版)
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
# 4. 最適化手法 (Predictive E-prop) バッチ対応版
# ==========================================
class EPropOptimizer:
    def __init__(self, cfg: Config, decay_r: float, decay_d: float, c_rd: float, device=device):
        self.cfg = cfg
        self.device = device
        self.decay_r = decay_r
        self.decay_d = decay_d
        self.c_rd = c_rd
        
        self.grad_w_rec_accumulated = torch.zeros((cfg.n_neurons, cfg.n_neurons), device=self.device)
        self.grad_w_out_accumulated = torch.zeros((cfg.n_outputs, cfg.n_neurons), device=self.device)
        self.grad_w_cls_accumulated = torch.zeros((cfg.n_classes, cfg.n_neurons), device=self.device)
        self.grad_b_cls_accumulated = torch.zeros(cfg.n_classes, device=self.device)

    def reset_traces(self, batch_size):
        self.batch_size = batch_size
        n = self.cfg.n_neurons
        self.e_trace = torch.zeros((batch_size, n, n), device=self.device)
        self.eps_b = torch.zeros((batch_size, n, n), device=self.device)
        self.q_trace = torch.zeros((batch_size, n, n), device=self.device)
        self.e_bar_bar = torch.zeros((batch_size, n, n), device=self.device)
        
        self.grad_w_rec_accumulated.zero_()
        self.grad_w_out_accumulated.zero_()
        self.grad_w_cls_accumulated.zero_()
        self.grad_b_cls_accumulated.zero_()
        
        self.spike_buffer = torch.zeros((batch_size, n), device=self.device)
        self.e_trace_sum_t_delays = torch.zeros((batch_size, n, n), device=self.device)
        self.step_counter = 0

    def update_traces_and_gradients(self, neurons, z_bar_bar_curr, z_bar_curr, z_bar_prev, z_bar_prev_prev, psi_prev, L_t_pred, d_t_pred, L_t_cls=None, d_t_cls=None):
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

        L_t_total = L_t_pred
            
        self.grad_w_rec_accumulated += torch.einsum('bi,bij->ij', L_t_total, self.e_bar_bar) / self.batch_size
        self.grad_w_out_accumulated += torch.einsum('bi,bj->ij', d_t_pred, z_bar_bar_curr) / self.batch_size
        
        if d_t_cls is not None:
            self.grad_w_cls_accumulated += torch.einsum('bi,bj->ij', d_t_cls, z_bar_bar_curr) / self.batch_size
            self.grad_b_cls_accumulated += torch.mean(d_t_cls, dim=0)

        all_spikes = torch.cat([g.z for g in neurons], dim=1)
        self.spike_buffer += all_spikes
        self.e_trace_sum_t_delays += self.e_trace
        self.step_counter += 1

        if self.step_counter >= self.cfg.t_delay:
            f_bar = self.spike_buffer / self.cfg.t_delay
            reg_signal = (self.cfg.lambda_reg / self.cfg.t_delay) * (f_bar - self.cfg.f_star)
            self.grad_w_rec_accumulated += torch.einsum('bi,bij->ij', reg_signal, self.e_trace_sum_t_delays) / self.batch_size
            
            self.spike_buffer.zero_()
            self.e_trace_sum_t_delays.zero_()
            self.step_counter = 0

    def apply_weight_update(self, w_rec, w_out, w_cls, b_cls):
        w_rec -= self.cfg.eta_rec * self.grad_w_rec_accumulated + self.cfg.eta_rec * (2.0 * self.cfg.lambda_w) * w_rec
        w_out -= self.cfg.eta_out * self.grad_w_out_accumulated + self.cfg.eta_out * (2.0 * self.cfg.lambda_w) * w_out
        w_cls -= self.cfg.eta_cls * self.grad_w_cls_accumulated.t()
        b_cls -= self.cfg.eta_cls * self.grad_b_cls_accumulated
        
        self.grad_w_rec_accumulated.zero_()
        self.grad_w_out_accumulated.zero_()
        self.grad_w_cls_accumulated.zero_()
        self.grad_b_cls_accumulated.zero_()
        
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

        self.w_cls = torch.randn(cfg.n_neurons, cfg.n_classes, device=device) / np.sqrt(cfg.n_neurons)
        self.b_cls = torch.zeros(cfg.n_classes, device=device)
        self.B_cls = torch.randn(cfg.n_neurons, cfg.n_classes, device=device) / np.sqrt(cfg.n_neurons)

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
# 6. テストデータ評価関数
# ==========================================
@torch.no_grad()
def evaluate_model(net, dataloader, cfg):
    total_loss_cls = 0.0
    total_correct = 0
    total_samples = 0
    
    for features_batch, labels_batch, mask_batch in dataloader:
        current_batch_size, max_time_steps, _ = features_batch.shape
        net.reset_state(batch_size=current_batch_size)
        
        batch_correct = 0
        batch_total = 0
        batch_loss = 0.0
        
        for t in range(max_time_steps):
            x_t = features_batch[:, t, :]
            label_t = labels_batch[:, t]
            m_t = mask_batch[:, t]
            
            all_z_bar_bar = torch.cat([net.lif.z_bar_bar, net.alif.z_bar_bar], dim=1)
            
            y_t = all_z_bar_bar @ net.w_out.t()
            d_t_pred = (y_t - x_t) * m_t.unsqueeze(1)
            
            logits_t = all_z_bar_bar @ net.w_cls + net.b_cls
            ce_loss = F.cross_entropy(logits_t, label_t, reduction='none') * m_t
            batch_loss += torch.mean(ce_loss).item()
            
            preds = torch.argmax(logits_t, dim=-1)
            correct = (preds == label_t).float() * m_t
            batch_correct += correct.sum().item()
            batch_total += m_t.sum().item()
            
            i_rec = all_z_bar_bar @ net.w_rec.t()
            i_in = d_t_pred @ net.w_in.t()
            i_fb = y_t @ net.w_fb.t()
            s_t = net.generate_ou_noise(current_batch_size)
            total_current = i_in + i_rec + i_fb + cfg.I_bias + s_t
            
            net.lif.step(total_current[:, :cfg.n_lif])
            net.alif.step(total_current[:, cfg.n_lif:])
            
        total_loss_cls += (batch_loss / max_time_steps) * current_batch_size
        total_correct += batch_correct
        total_samples += batch_total
        
    avg_loss = total_loss_cls / max(len(dataloader.dataset), 1)
    accuracy = total_correct / max(total_samples, 1) * 100.0
    return avg_loss, accuracy


# ==========================================
# 7. 学習実行ルーチン
# ==========================================
def train_timit_pickle(processed_dir="../datasets/timit_processed", feature_type="mfcc", batch_size=8, epochs=30):
    train_dir = os.path.join(processed_dir, 'train')
    dev_dir = os.path.join(processed_dir, 'develop')
    test_dir = os.path.join(processed_dir, 'test')

    print(f"Loading datasets from: {processed_dir}")
    train_dataset = TimitPickleDataset(train_dir, feature_type=feature_type)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    test_loader = None
    if os.path.exists(test_dir):
        test_dataset = TimitPickleDataset(test_dir, feature_type=feature_type)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        print(f"Test utterances: {len(test_dataset)}, Test batches: {len(test_loader)}")

    print(f"Train utterances: {len(train_dataset)}, Train batches: {len(train_loader)}")

    # 入力次元の自動取得 (MFCC: 39, FBank: 123)
    sample_feat, _ = train_dataset[0]
    n_features = sample_feat.shape[1]

    cfg = Config()
    cfg.n_inputs = n_features
    cfg.n_outputs = n_features
    cfg.epochs = epochs
    cfg.batch_size = batch_size

    net = PredictiveEPropNet(cfg=cfg, device=device)

    for epoch in range(epochs):
        epoch_loss_pred = 0.0
        epoch_loss_cls = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (features_batch, labels_batch, mask_batch) in enumerate(train_loader):
            current_batch_size, max_time_steps, _ = features_batch.shape

            net.reset_state(batch_size=current_batch_size)

            z_bar_prev = torch.zeros((current_batch_size, cfg.n_neurons), device=device)
            z_bar_prev_prev = torch.zeros((current_batch_size, cfg.n_neurons), device=device)
            psi_prev = torch.zeros((current_batch_size, cfg.n_neurons), device=device)

            batch_loss_pred = 0.0
            batch_loss_cls = 0.0
            batch_correct = 0
            batch_total = 0

            for t in range(max_time_steps):
                x_t = features_batch[:, t, :] 
                label_t = labels_batch[:, t] 
                m_t = mask_batch[:, t] 

                all_z_bar_bar = torch.cat([net.lif.z_bar_bar, net.alif.z_bar_bar], dim=1)
                all_z_bar = torch.cat([net.lif.z_bar, net.alif.z_bar], dim=1)

                # 1. 予測出力と予測誤差
                y_t = all_z_bar_bar @ net.w_out.t()
                d_t_pred = (y_t - x_t) * m_t.unsqueeze(1)
                batch_loss_pred += torch.mean(d_t_pred ** 2).item()

                # 2. 分類出力と分類誤差
                logits_t = all_z_bar_bar @ net.w_cls + net.b_cls
                ce_loss = F.cross_entropy(logits_t, label_t, reduction='none') * m_t
                batch_loss_cls += torch.mean(ce_loss).item()

                preds = torch.argmax(logits_t, dim=-1)
                correct = (preds == label_t).float() * m_t
                batch_correct += correct.sum().item()
                batch_total += m_t.sum().item()

                probs = F.softmax(logits_t, dim=-1)
                d_t_cls = probs.clone()
                d_t_cls[torch.arange(current_batch_size), label_t] -= 1.0
                d_t_cls = d_t_cls * m_t.unsqueeze(1)

                i_rec = all_z_bar_bar @ net.w_rec.t()
                i_in = d_t_pred @ net.w_in.t()
                i_fb = y_t @ net.w_fb.t()
                s_t = net.generate_ou_noise(current_batch_size)

                total_current = i_in + i_rec + i_fb + cfg.I_bias + s_t

                net.lif.step(total_current[:, :cfg.n_lif])
                net.alif.step(total_current[:, cfg.n_lif:])

                L_t_pred = d_t_pred @ net.B_pred.t()
                L_t_cls = d_t_cls @ net.B_cls.t()
                all_psi = torch.cat([net.lif.psi, net.alif.psi], dim=1)

                net.optimizer.update_traces_and_gradients(
                    net.neurons, all_z_bar_bar, all_z_bar, z_bar_prev, z_bar_prev_prev, psi_prev, 
                    L_t_pred, d_t_pred, L_t_cls, d_t_cls
                )

                z_bar_prev_prev = z_bar_prev.clone()
                z_bar_prev = all_z_bar.clone()
                psi_prev = all_psi.clone()

            # 勾配集計と更新
            net.optimizer.grad_w_rec_accumulated /= max_time_steps
            net.optimizer.grad_w_out_accumulated /= max_time_steps
            net.optimizer.grad_w_cls_accumulated /= max_time_steps
            net.optimizer.grad_b_cls_accumulated /= max_time_steps

            clip_val = 1.0
            torch.clamp_(net.optimizer.grad_w_rec_accumulated, -clip_val, clip_val)
            torch.clamp_(net.optimizer.grad_w_out_accumulated, -clip_val, clip_val)
            torch.clamp_(net.optimizer.grad_w_cls_accumulated, -clip_val, clip_val)
            torch.clamp_(net.optimizer.grad_b_cls_accumulated, -clip_val, clip_val)

            net.w_rec, net.w_out, net.w_cls, net.b_cls = net.optimizer.apply_weight_update(
                net.w_rec, net.w_out, net.w_cls, net.b_cls
            )
            net.w_rec *= net.w_rec_mask

            epoch_loss_pred += batch_loss_pred / max_time_steps
            epoch_loss_cls += batch_loss_cls / max_time_steps
            epoch_correct += batch_correct
            epoch_total += batch_total

            if batch_idx % 100 == 0:
                b_acc = batch_correct / max(batch_total, 1) * 100.0
                print(f"  Train Batch {batch_idx}/{len(train_loader)} | Cls Loss: {batch_loss_cls/max_time_steps:.4f} | Acc: {b_acc:.2f}%")

        epoch_acc = epoch_correct / max(epoch_total, 1) * 100.0
        print(f"=== Epoch {epoch+1} Train Summary | Pred Loss: {epoch_loss_pred/len(train_loader):.4f} | Cls Loss: {epoch_loss_cls/len(train_loader):.4f} | Accuracy: {epoch_acc:.2f}% ===")

        if test_loader is not None:
            test_loss, test_acc = evaluate_model(net, test_loader, cfg)
            print(f"*** Epoch {epoch+1} Test Evaluation  | Test Cls Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}% ***\n")


if __name__ == "__main__":
    # 前処理スクリプトで出力したディレクトリを指定
    PROCESSED_TIMIT_PATH = "/home/keitaro-sunagawa/Master-Research/Code/Predictive_E-prop/mainTask"
    
    train_timit_pickle(
        processed_dir=PROCESSED_TIMIT_PATH,
        feature_type="mfcc", # 'mfcc' または 'fbank'
        batch_size=8,
        epochs=30
    )