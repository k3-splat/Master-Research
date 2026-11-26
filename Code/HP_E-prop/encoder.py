#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIMIT Preprocessing and Dataset Pipeline based on IGITUGraz/eligibility_propagation
Compatible with PyTorch models (Predictive E-prop / SNN).
"""

import os
import csv
import json
import pickle
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import scipy.io.wavfile as wavfile
from sphfile import SPHFile
from python_speech_features import mfcc, fbank
import torch
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 音素マッピング定義 (Graves 39-phoneme reduction)
# ==========================================
PHONEME_REDUCTION_TABLE: Dict[str, str] = {
    'ao': 'aa',
    'ax': 'ah',
    'ax-h': 'ah',
    'axr': 'er',
    'hv': 'hh',
    'ix': 'ih',
    'el': 'l',
    'em': 'm',
    'en': 'n',
    'nx': 'n',
    'eng': 'ng',
    'zh': 'sh',
    'pcl': 'sil',
    'tcl': 'sil',
    'kcl': 'sil',
    'bcl': 'sil',
    'dcl': 'sil',
    'gcl': 'sil',
    'h#': 'sil',
    'pau': 'sil',
    'epi': 'sil',
    'ux': 'uw',
    'q': 'sil',
}

REDUCED_PHN_LIST: List[str] = [
    'sil', 'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'er',
    'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy',
    'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z'
]
PHN_TO_ID: Dict[str, int] = {phn: i for i, phn in enumerate(REDUCED_PHN_LIST)}

DEVELOP_SELECTION: List[str] = [
    'dr2/mgwt0', 'dr2/mpdf0', 'dr2/mjar0', 'dr2/mmdm2', 'dr2/mmdb1',
    'dr4/fgjd0', 'dr4/fjmg0', 'dr4/fadg0', 'dr4/fnmr0', 'dr4/frew0',
    'dr4/mteb0', 'dr4/fedw0', 'dr4/mdls0', 'dr4/fsem0', 'dr4/fdms0',
    'dr4/mroa0', 'dr4/mbns0', 'dr5/fcal1', 'dr5/fmah0', 'dr5/mrws1',
    'dr8/fjsj0', 'dr8/majc0', 'dr6/fdrw0', 'dr6/mjfc0', 'dr6/mrjr0',
    'dr7/mdlf0', 'dr7/mers0', 'dr7/fmml0', 'dr7/mrcs0', 'dr7/mrjm4',
    'dr7/mdvc0', 'dr1/faks0', 'dr1/fdac1', 'dr1/mjsw0', 'dr1/mreb0',
    'dr1/fjem0', 'dr3/mmwh0', 'dr3/mglb0', 'dr3/fkms0', 'dr3/mgjf0',
    'dr3/mtdt0', 'dr3/mrtk0', 'dr3/mtaa0', 'dr3/mcsh0', 'dr3/mmjr0',
    'dr3/mwjg0', 'dr3/mbdg0', 'dr3/fcmh0', 'dr3/mbwm0', 'dr3/mthc0'
]

CORE_TEST_SELECTION: List[str] = [
    'dr2/fpas0', 'dr2/mtas1', 'dr2/mwew0', 'dr4/mtls0', 'dr4/mlll0',
    'dr4/fjlm0', 'dr5/fnlp0', 'dr5/mbpm0', 'dr5/mklt0', 'dr8/fmld0',
    'dr8/mpam0', 'dr8/mjln0', 'dr6/mcmj0', 'dr6/fmgd0', 'dr6/mjdh0',
    'dr7/fdhc0', 'dr7/mnjm0', 'dr7/mgrt0', 'dr1/mdab0', 'dr1/felc0',
    'dr1/mwbt0', 'dr3/fpkt0', 'dr3/mjmp0', 'dr3/mlnt0'
]


# ==========================================
# 2. 前処理ユーティリティ関数
# ==========================================
def add_derivatives(feature: np.ndarray) -> np.ndarray:
    """1次微係数(Delta)および2次微係数(Delta-Delta)を計算して連結する"""
    def get_delta(v: np.ndarray) -> np.ndarray:
        delta = np.zeros_like(v)
        delta[1:-1] = v[2:] - v[:-2]
        return delta

    d_feat = get_delta(feature)
    d2_feat = get_delta(d_feat)
    return np.concatenate([feature, d_feat, d2_feat], axis=1)


def process_audio(
    sphere_file_path: str,
    tmp_path: str = "/tmp/timit_processing",
    window_step_sec: float = 0.01,
    window_size_sec: float = 0.025,
    n_filter_bank: int = 40
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """SPH/WAVファイルを読み込み、MFCCおよびFilter Bank特徴量を抽出する"""
    os.makedirs(tmp_path, exist_ok=True)
    basename = os.path.basename(sphere_file_path)
    wav_readable_path = os.path.join(tmp_path, basename) + "_readable.wav"

    try:
        sph = SPHFile(sphere_file_path)
        sph.write_wav(wav_readable_path)
        fs, wav = wavfile.read(wav_readable_path)
    except Exception:
        fs, wav = wavfile.read(sphere_file_path)
    finally:
        if os.path.exists(wav_readable_path):
            os.remove(wav_readable_path)

    mfccs = mfcc(wav, fs, winstep=window_step_sec, winlen=window_size_sec)
    fbs, energy = fbank(wav, fs, nfilt=n_filter_bank, winstep=window_step_sec, winlen=window_size_sec)
    fbs_with_energy = np.concatenate([fbs, energy[:, None]], axis=1)

    return mfccs, fbs_with_energy, wav, fs


def process_phn(
    phn_path: str,
    num_windows: int,
    window_step_samples: int = 160
) -> np.ndarray:
    """音素アライメントファイル(.phn)を読み込み、フレーム単位の縮約音素ID系列を作成する"""
    vector = np.zeros(num_windows, dtype=np.int64)
    with open(phn_path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if len(row) < 3:
                continue
            k_start = int(row[0]) // window_step_samples
            k_end = int(row[1]) // window_step_samples
            raw_phn = row[2].lower()

            mapped_phn = PHONEME_REDUCTION_TABLE.get(raw_phn, raw_phn)
            phn_id = PHN_TO_ID.get(mapped_phn, PHN_TO_ID['sil'])
            vector[k_start:k_end] = phn_id
    return vector


def find_path_case_insensitive(base_path: str, target: str) -> Optional[str]:
    """大文字/小文字の違いを吸収してパスを探索する"""
    if not os.path.isdir(base_path):
        return None
    for entry in os.listdir(base_path):
        if entry.lower() == target.lower():
            return os.path.join(base_path, entry)
    return None


# ==========================================
# 3. TIMIT データセット前処理実行パイプライン
# ==========================================
def preprocess_raw_timit(
    raw_data_path: str,
    output_path: str,
    exclude_sa: bool = True,
    window_step_sec: float = 0.01,
    window_size_sec: float = 0.025,
    sampling_rate: int = 16000
):
    """
    TIMITコーパス全体を走査し、MFCC / FilterBank / 音素ラベルを抽出し、
    訓練セットの平均・標準偏差で正規化して保存する。
    """
    window_step_samples = int(window_step_sec * sampling_rate)
    splits = [
        ('train', 'train'),
        ('test', 'develop'),
        ('test', 'test')
    ]
    drs = [f'dr{i}' for i in range(1, 9)]

    processed_data = {
        'train': {'mfcc': [], 'fbank': [], 'phn': [], 'wav': []},
        'develop': {'mfcc': [], 'fbank': [], 'phn': [], 'wav': []},
        'test': {'mfcc': [], 'fbank': [], 'phn': [], 'wav': []}
    }

    print(f"Starting TIMIT preprocessing from: {raw_data_path}")

    for dataset_source, dataset_target in splits:
        print(f"Extracting features for target: {dataset_target} (source: {dataset_source})")
        src_dir = find_path_case_insensitive(raw_data_path, dataset_source)
        if src_dir is None:
            print(f"Warning: Directory '{dataset_source}' not found under '{raw_data_path}'")
            continue

        for dr in drs:
            dr_path = find_path_case_insensitive(src_dir, dr)
            if dr_path is None or not os.path.isdir(dr_path):
                continue

            for speaker in os.listdir(dr_path):
                speaker_ref = f"{dr.lower()}/{speaker.lower()}"
                if dataset_source == 'train':
                    use_speaker = True
                elif dataset_target == 'develop':
                    use_speaker = (speaker_ref in DEVELOP_SELECTION)
                elif dataset_target == 'test':
                    use_speaker = (speaker_ref in CORE_TEST_SELECTION)
                else:
                    use_speaker = False

                if not use_speaker:
                    continue

                speaker_path = os.path.join(dr_path, speaker)
                for f in os.listdir(speaker_path):
                    if not f.lower().endswith('.wav'):
                        continue
                    if exclude_sa and f.lower().startswith('sa'):
                        continue

                    stem = os.path.splitext(f)[0]
                    wav_full_path = os.path.join(speaker_path, f)
                    
                    phn_filename = None
                    for entry in os.listdir(speaker_path):
                        if entry.lower() == f"{stem.lower()}.phn":
                            phn_filename = entry
                            break

                    if phn_filename is None:
                        continue

                    phn_full_path = os.path.join(speaker_path, phn_filename)

                    mfccs, fbanks, wav, _ = process_audio(
                        wav_full_path,
                        window_step_sec=window_step_sec,
                        window_size_sec=window_size_sec
                    )

                    phns = process_phn(
                        phn_full_path,
                        num_windows=mfccs.shape[0],
                        window_step_samples=window_step_samples
                    )

                    mfccs = add_derivatives(mfccs)
                    fbanks = add_derivatives(fbanks)

                    assert len(phns) == mfccs.shape[0]

                    processed_data[dataset_target]['mfcc'].append(mfccs.astype(np.float32))
                    processed_data[dataset_target]['fbank'].append(fbanks.astype(np.float32))
                    processed_data[dataset_target]['phn'].append(phns.astype(np.int64))
                    processed_data[dataset_target]['wav'].append(wav)

    if len(processed_data['train']['mfcc']) == 0:
        raise RuntimeError("No training samples processed. Please check the dataset directory structure.")

    print("Computing mean and standard deviation over training set...")
    train_mfccs_concat = np.concatenate(processed_data['train']['mfcc'], axis=0)
    train_fbanks_concat = np.concatenate(processed_data['train']['fbank'], axis=0)

    stats = {
        'mfcc_mean': np.mean(train_mfccs_concat, axis=0),
        'mfcc_std': np.std(train_mfccs_concat, axis=0) + 1e-10,
        'fbank_mean': np.mean(train_fbanks_concat, axis=0),
        'fbank_std': np.std(train_fbanks_concat, axis=0) + 1e-10
    }

    for target in ['train', 'develop', 'test']:
        processed_data[target]['mfcc'] = [
            (feat - stats['mfcc_mean']) / stats['mfcc_std'] for feat in processed_data[target]['mfcc']
        ]
        processed_data[target]['fbank'] = [
            (feat - stats['fbank_mean']) / stats['fbank_std'] for feat in processed_data[target]['fbank']
        ]

    os.makedirs(output_path, exist_ok=True)
    for target in ['train', 'develop', 'test']:
        target_dir = os.path.join(output_path, target)
        os.makedirs(target_dir, exist_ok=True)

        with open(os.path.join(target_dir, 'mfccs.pickle'), 'wb') as f:
            pickle.dump(processed_data[target]['mfcc'], f, protocol=4)
        with open(os.path.join(target_dir, 'filter_banks.pickle'), 'wb') as f:
            pickle.dump(processed_data[target]['fbank'], f, protocol=4)
        with open(os.path.join(target_dir, 'reduced_phonems.pickle'), 'wb') as f:
            pickle.dump(processed_data[target]['phn'], f, protocol=4)
        with open(os.path.join(target_dir, 'wav.pickle'), 'wb') as f:
            pickle.dump(processed_data[target]['wav'], f, protocol=4)

    with open(os.path.join(output_path, 'reduced_phonem_list.json'), 'w') as f:
        json.dump(REDUCED_PHN_LIST, f, indent=4)
    with open(os.path.join(output_path, 'norm_stats.pickle'), 'wb') as f:
        pickle.dump(stats, f, protocol=4)

    print(f"Preprocessed TIMIT dataset saved successfully to '{output_path}'")


# ==========================================
# 4. PyTorch Dataset & Collate Fn
# ==========================================
class TimitProcessedDataset(Dataset):
    def __init__(self, processed_dir: str, feature_type: str = 'mfcc'):
        assert feature_type in ['mfcc', 'fbank'], "feature_type must be 'mfcc' or 'fbank'"
        self.feature_file = 'mfccs.pickle' if feature_type == 'mfcc' else 'filter_banks.pickle'
        
        feat_path = os.path.join(processed_dir, self.feature_file)
        phn_path = os.path.join(processed_dir, 'reduced_phonems.pickle')

        with open(feat_path, 'rb') as f:
            self.features = pickle.load(f)
        with open(phn_path, 'rb') as f:
            self.labels = pickle.load(f)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return feat, label


def timit_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], device: torch.device):
    features, labels = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)

    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PHN_TO_ID['sil'])

    mask = torch.arange(padded_features.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return padded_features.to(device), padded_labels.to(device), mask.float().to(device)


# ==========================================
# 5. スクリプト実行部
# ==========================================
if __name__ == '__main__':
    # 手元の生データ配置パスと出力先パスを指定
    raw_timit_dir = '/home/keitaro-sunagawa/Master-Research/TIMIT'
    processed_timit_dir = '/home/keitaro-sunagawa/Master-Research/Code/Predictive_E-prop/mainTask'

    # 前処理の実行（未作成の場合）
    if not os.path.exists(os.path.join(processed_timit_dir, 'train', 'mfccs.pickle')):
        if os.path.exists(raw_timit_dir):
            preprocess_raw_timit(raw_timit_dir, processed_timit_dir)
        else:
            print(f"Raw TIMIT path '{raw_timit_dir}' not found. Please verify the directory path.")

    # PyTorch DataLoader 動作確認
    train_pickle_path = os.path.join(processed_timit_dir, 'train')
    if os.path.exists(os.path.join(train_pickle_path, 'mfccs.pickle')):
        dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        train_set = TimitProcessedDataset(train_pickle_path, feature_type='mfcc')
        loader = DataLoader(
            train_set,
            batch_size=8,
            shuffle=True,
            collate_fn=lambda b: timit_collate_fn(b, dev)
        )

        feats, lbls, msk = next(iter(loader))
        print(f"DataLoader verification succeeded:")
        print(f"  Features shape: {feats.shape} (Batch, Time, Features [39 for MFCC])")
        print(f"  Labels shape:   {lbls.shape}")
        print(f"  Mask shape:     {msk.shape}")