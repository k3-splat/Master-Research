%% Encode a single audio file from the Podcast ECoG dataset using BAE
clear all; clc; dbstop if error;

% 1. BAEツール群へのパスを追加（ご自身の環境に合わせて絶対パスまたは相対パスで指定）
addpath('/home/keitaro-sunagawa/Master-Research/Code/Encoder/tools'); 
% addpath('G:\Resarch Work\Threshold code and temporal masking'); 

% 2. エンコードしたい特定の音声ファイルのパスを直接指定
target_file_path = '/home/keitaro-sunagawa/Master-Research/ds005574/stimuli/podcast.wav'; 

disp(['Processing file: ', target_file_path]);

try
    % 音声ファイルの読み込み
    [audio_data, fs] = audioread(target_file_path);
    
    % 【追加】ステレオ（2チャンネル以上）の場合は平均をとってモノラルに変換
    if size(audio_data, 2) > 1
        audio_data = mean(audio_data, 2);
    end
    
    % BAEの推奨設定に合わせて、サンプリングレートを16kHzに変換（必要な場合）
    if fs ~= 16000
        audio_data = resample(audio_data, 16000, fs);
    end
    
    % ===============================================================
    % 3. BAEエンコード処理の実行
    % ===============================================================
    [spike_pattern, binary_pattern] = encode_single_wav(audio_data);
    
    % 4. 結果を.matファイルに保存
    [~, file_name, ~] = fileparts(target_file_path);
    name1 = [file_name, '_spike_timing.mat'];
    name2 = [file_name, '_binary_matrix.mat'];
    
    % スパイクタイミングとバイナリ行列をそれぞれ保存
    save(name1, 'spike_pattern');
    save(name2, 'binary_pattern');
    disp(['Encoding completed. Saved to ', name1, ' and ', name2]);
    
catch ME
    disp(['Error processing ', target_file_path, ': ', ME.message]);
end