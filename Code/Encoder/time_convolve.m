function parallel_stream = time_convolve(input_wav,wavelet_len)

load('filter_group1.mat');

Ls=length(input_signal);
N_channel=20;
parallel_stream=zeros(N_channel,Ls);

    % time domian convolution
    for i=1:1:N_channel
        filter=cell2mat(filter_group(1,wavelet_len,i));
        temp=conv(input_wav,filter,'same');  
        parallel_stream(i,:)=temp; 
    end


end
