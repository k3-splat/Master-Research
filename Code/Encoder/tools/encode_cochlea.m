function encode_result=encode_cochlea(signal,wavelet_window)

%% read out one file for encoding and decoding

Ly=length(signal);
% load('Filter_data_200');
% filter_wavelets=filter_wave(2:end,:);


%load('filter_cutoff60.mat');
%filter_wavelets=filter_cutoff(2:end,:);
%% Encoding the speech signal
encode_result=encoding_convolution(signal,wavelet_window);

end