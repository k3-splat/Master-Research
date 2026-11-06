import numpy as np
import os
import csv
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ---------------------------------------------------------
# 設定 (Spike-TIMIT向けに変更)
# ---------------------------------------------------------
DATA_PATH = '/home/keitaro-sunagawa/Master-Research/Spike_TIMIT/Spike TIMIT'
TMP_PATH = '/tmp/timit_processing'
OUTPUT_PATH = '../datasets/spike_timit_processed'

interactive_plot_histogram = True
exclude_sa_files = True
normalize_over_training_set = False # スパイクデータのため正規化は無効化

n_drs = 8
SAMPLING_RATE = 16000
# 16000Hzの音声に対するbinary.matのタイムビン幅（サンプル数）
# 例: binary.matが1msごとのスパイクなら 16サンプル/ms
samples_per_time_bin = 16 

dataset_sources = ['train','test', 'test']
dataset_targets = ['train','test','develop']
drs = ['dr'+str(i) for i in range(1,n_drs +1)]

reduced_phn_list = ['sil']
phn_list = ['h#']
word_list = [None]

develop_selection = [
'dr2/mgwt0','dr2/mpdf0','dr2/mjar0','dr2/mmdm2','dr2/mmdb1',
'dr4/fgjd0','dr4/fjmg0','dr4/fadg0','dr4/fnmr0','dr4/frew0',
'dr4/mteb0','dr4/fedw0','dr4/mdls0','dr4/fsem0','dr4/fdms0',
'dr4/mroa0','dr4/mbns0','dr5/fcal1','dr5/fmah0','dr5/mrws1',
'dr8/fjsj0','dr8/majc0','dr6/fdrw0','dr6/mjfc0','dr6/mrjr0',
'dr7/mdlf0','dr7/mers0','dr7/fmml0','dr7/mrcs0','dr7/mrjm4',
'dr7/mdvc0','dr1/faks0','dr1/fdac1','dr1/mjsw0','dr1/mreb0',
'dr1/fjem0','dr3/mmwh0','dr3/mglb0','dr3/fkms0','dr3/mgjf0',
'dr3/mtdt0','dr3/mrtk0','dr3/mtaa0','dr3/mcsh0','dr3/mmjr0',
'dr3/mwjg0','dr3/mbdg0','dr3/fcmh0','dr3/mbwm0','dr3/mthc0',]

core_test_selection = [
'dr2/fpas0','dr2/mtas1','dr2/mwew0','dr4/mtls0','dr4/mlll0',
'dr4/fjlm0','dr5/fnlp0','dr5/mbpm0','dr5/mklt0','dr8/fmld0',
'dr8/mpam0','dr8/mjln0','dr6/mcmj0','dr6/fmgd0','dr6/mjdh0',
'dr7/fdhc0','dr7/mnjm0','dr7/mgrt0','dr1/mdab0','dr1/felc0',
'dr1/mwbt0','dr3/fpkt0','dr3/mjmp0','dr3/mlnt0',]

phonem_reduction_table = {
    'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er',
    'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm',
    'en': 'n', 'nx': 'n', 'eng': 'ng', 'zh': 'sh',
    'pcl': 'sil', 'tcl': 'sil', 'kcl': 'sil', 'bcl': 'sil',
    'dcl': 'sil', 'gcl': 'sil', 'h#': 'sil', 'pau': 'sil',
    'epi': 'sil', 'ux': 'uw', 'q': 'sil',
}

def get_phn_or_word_id(phn_or_word,phn_or_word_list,freeze_list):
    inds = np.where(np.array(phn_or_word_list) == phn_or_word)[0]
    assert len(inds) in [0,1]
    if len(inds) == 1:
        return inds[0]
    if freeze_list:
        print('WARNING: List should be frozen (test set) but phn or word \'{}\' is not in list {}.'.format(phn_or_word,phn_or_word_list))
    phn_or_word_list.append(phn_or_word)
    return len(phn_or_word_list) - 1

def get_file_name_tuple_from_speaker_path(speaker_path):
    files = []
    file_names = os.listdir(speaker_path)
    assert(file_names), 'Speaker path: {} is empty ({})'.format(speaker_path,file_names)
    for f in file_names:
        if f.endswith('.mat'):
            trunc = f[:-4]
            if f.endswith('_binary_matrix.mat'):
                trunc = f[:-11]
            if not(exclude_sa_files) or not f.startswith('sa'):
                mat_file = f
                phn_file = trunc + '.phn'
                wrd_file = trunc + '.wrd'
                txt_file = trunc + '.txt'
                files.append((mat_file, phn_file, wrd_file, txt_file))
    return files

def process_mat(path, mat_file):
    assert(mat_file.endswith('.mat')), 'Wrong file name, should be a .mat: {}'.format(mat_file)
    mat_path = os.path.join(path, mat_file)
    mat_data = loadmat(mat_path)
    
    # 適切なキー名に変更してください
    spikes = mat_data.get('spikes')
    if spikes is None:
        # 'spikes'が存在しない場合、キーを検索
        for k in mat_data.keys():
            if not k.startswith('__'):
                spikes = mat_data[k]
                break
                
    # ネットワークは (Time, Channels) を想定しているため形状を調整
    if spikes.shape[0] < spikes.shape[1] and spikes.shape[0] < 200:
        spikes = spikes.T
    return spikes

def process_phn_or_word(path, phn_or_word_file, meta_data, reduce_phonem):
    vector = np.zeros(meta_data['num_windows'],dtype=int)
    path = os.path.join(path, phn_or_word_file)
    is_phn = True if path[-4:] == '.phn' else False
    freeze_ids = meta_data['dataset_source'] in ['test','develop'] and is_phn

    if os.path.exists(path):
        with open(path,'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                row = list(row)
                k_start = int(row[0]) // samples_per_time_bin
                k_end = int(row[1]) // samples_per_time_bin
                
                # スパイク長を超過しないように補正
                k_end = min(k_end, meta_data['num_windows'])
                k_start = min(k_start, meta_data['num_windows'] - 1)
                
                phn_or_word = row[2]
                if is_phn:
                    if reduce_phonem:
                        if phn_or_word in phonem_reduction_table.keys():
                            phn_or_word = phonem_reduction_table[phn_or_word]
                        ind = get_phn_or_word_id(phn_or_word,reduced_phn_list,freeze_ids)
                        vector[k_start:k_end] = ind
                        assert(ind < 39),'Found reduced index {} should be lower than 39.'.format(ind)
                    else:
                        vector[k_start:k_end] = get_phn_or_word_id(phn_or_word,phn_list,freeze_ids)
                else:
                    vector[k_start:k_end] = get_phn_or_word_id(phn_or_word,word_list,freeze_ids)
    return vector

def process_txt(path,txt_file):
    txt_path = os.path.join(path,txt_file)
    start_time, end_time, sentence = 0, 0, ""
    if os.path.exists(txt_path):
        with open(txt_path,'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                row = list(row)
                start_time = int(row[0])
                end_time = int(row[1])
                sentence = ' '.join(row[2:])
                break
    return start_time,end_time,sentence

if __name__ == '__main__':
    if interactive_plot_histogram:
        plt.ion()
        fig, axes = plt.subplots(3, figsize=(5, 6))

    for dataset_source,dataset_target in zip(dataset_sources,dataset_targets):
        print('Processing {} dialects for dataset: '.format(n_drs) + dataset_target)

        metadata_stack = []
        spike_stack = []
        phn_stack = []
        reduced_phn_stack = []
        word_stack = []
        dataset_file_count = 0

        for dr in drs:
            dialect_file_count = 0
            dr_path = os.path.join(DATA_PATH,dataset_source,dr)
            full_speaker_path_list = os.listdir(dr_path)
            selected_speaker_path_list = []
            
            for speaker in full_speaker_path_list:
                speaker_reference = dr + '/' + speaker
                use_speaker = False
                if dataset_source == 'train': use_speaker=True
                if speaker_reference in develop_selection and dataset_target == 'develop': use_speaker=True
                if speaker_reference in core_test_selection and dataset_target == 'test': use_speaker=True

                if use_speaker:
                    selected_speaker_path_list.append(speaker)

            for speaker in selected_speaker_path_list:
                speaker_path = os.path.join(dr_path,speaker)

                for mat_file,phn_file,wrd_file,txt_file in get_file_name_tuple_from_speaker_path(speaker_path):
                    dialect_file_count += 1
                    sample_start, sample_end, text = process_txt(speaker_path, txt_file)

                    # process_wav の代わりに process_mat を呼び出す
                    spikes = process_mat(speaker_path, mat_file)

                    duration = (sample_end - sample_start) / SAMPLING_RATE
                    meta_data = {'dr': dr,
                                 'speaker': speaker,
                                 'sample_start': sample_start,
                                 'sample_end': sample_end,
                                 'text': text,
                                 'dataset_source': dataset_source,
                                 'dataset_target': dataset_target,
                                 'duration': duration,
                                 'num_windows': spikes.shape[0]
                                }

                    phns = process_phn_or_word(speaker_path,phn_file,meta_data,False)
                    reduced_phns = process_phn_or_word(speaker_path,phn_file,meta_data,True)
                    words = process_phn_or_word(speaker_path,wrd_file,meta_data,False)

                    metadata_stack.append(meta_data)
                    spike_stack.append(spikes)
                    phn_stack.append(phns)
                    reduced_phn_stack.append(reduced_phns)
                    word_stack.append(words)

            print('Dialect {}: {} speakers - {} audio/label file pairs'.format(dr,len(selected_speaker_path_list),dialect_file_count))
            dataset_file_count += dialect_file_count

            if interactive_plot_histogram and dataset_file_count > 0:
                ax, ax2, ax3 = axes
                tfrom, tto = 5, 60
                ax.clear()
                i = len(metadata_stack)-1
                plot_spikes = spike_stack[i][tfrom:tto, :].T
                ax.imshow(plot_spikes, aspect="auto", cmap='Greys')
                ax.set_title(metadata_stack[i]['text'])
                plt.draw()
                plt.pause(0.1)

        print('Processed {} audio/label file pairs for dataset {} '.format(dataset_file_count,dataset_target))

        assert(len(reduced_phn_list) <= 40),'Encountered {} phonems in the dataset'.format(len(reduced_phn_list))

        print('Writing the processed data in pickle and json files.')
        if not(Path(OUTPUT_PATH).is_dir()):
            os.mkdir(OUTPUT_PATH)
        output_dataset_path = os.path.join(OUTPUT_PATH, dataset_target)
        if not(Path(output_dataset_path).is_dir()):
            os.mkdir(output_dataset_path)

        for stack,file_name in zip(
                [metadata_stack,spike_stack,phn_stack,reduced_phn_stack,word_stack],
                ['metadata.pickle','spike_mat.pickle','phonems.pickle','reduced_phonems.pickle','words.pickle']):
            with open(os.path.join(output_dataset_path, file_name), 'wb') as f:
                pickle.dump(stack,f,protocol=4)

        def find_reduced_phn_id(phn):
            if phn in phonem_reduction_table.keys():
                return reduced_phn_list.index(phonem_reduction_table[phn])
            else:
                return reduced_phn_list.index(phn)

        reduced_phn_index_mapping = [find_reduced_phn_id(phn) for phn in phn_list]

        for label_list,file_name in zip([word_list,phn_list,reduced_phn_list,reduced_phn_index_mapping],
                                        ['word_list.json','phonem_list.json','reduced_phonem_list.json','reduced_phn_index_mapping.json']):
            with open(os.path.join(output_dataset_path, file_name), 'w') as f:
                json.dump(label_list,f,indent=4)