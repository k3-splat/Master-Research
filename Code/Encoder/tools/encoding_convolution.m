function Channel_signal=encoding_convolution(input_signal,wavelet_window)

load('filter_group1.mat');
%load('Bark_filter_20.mat');

%step=window/2;
Ls=length(input_signal);
%L=ceil(Ls/step);
N_channel=20;
Channel_signal=zeros(N_channel,Ls);
%center=ceil((Ls+Lw-1)/2);

for i=1:1:N_channel

   %filter=time_domain_matrix(i,:);
    filter=cell2mat(filter_group(1,wavelet_window,i));
    temp=conv(input_signal,filter,'same');
  
    Channel_signal(i,:)=temp;
    
    
    %Channel_signal(i,:)=temp(1:step:Ls)*length_c(i);
    
end

end