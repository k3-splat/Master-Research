% time-domain coclhear filter bank
% IJCNN 2018 Pan Zihan

% input_wav: the input sound wave, read from .wav file
% fs: sampling frequency, read from .wav file 
% wavelet_len: index of wavelet length, optimal at 4
% window_len: framing winodw length, typically 30ms
% parallel_stream:  time-domain signal streams, parallel in cochlear sub-band
% cochlea_spectro: the output spectrogram, 2-D (time dimension, sub-band dimension)

%%
% read sound wave
[input_wav,fs]= audioread('sa1.wav');

% time domain cochlear filter convolution
wavelet_len=4; % wavelet length index 
parallel_stream = time_convolve(input_wav,wavelet_len);

% log-scale energy framing
window_len=30*fs/1000; % 30ms framing window length
cochlea_spectro=log_energy(parallel_stream,window_len);




 