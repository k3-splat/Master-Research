function [output,binary]=encode_single_wav_NoMask(input_wav)

addpath('\tools');

wavelet_window=4;
%iteration=5;
N_train=1;

%window_vector=[240,320,400,480,560,640,720,800,880,960,1040,1120];  %no 480
%window_vector=[800];
window_length=160;


process_train=cell(N_train,1);


for i=1:N_train

    encode_train=encode_cochlea(input_wav,wavelet_window); %normalize them?
   
    process_train{i}=average_withMasking(encode_train,window_length);

      
end


 signal_train=global_normalization(process_train);  % input normalized spectrogram

 N_channel=20;

nAfferents=N_channel*(2*15+1);
ptnTrain=cell(N_train,nAfferents);
TmaxTrain=zeros(N_train,1);
for i=1:1:N_train
    
    sample=signal_train{i};


   [output,T]=threshold_code(sample);
   TmaxTrain(i)=T;
    
    
    for j=1:1:nAfferents
        ptnTrain{i,j}=output{j};
    end

end


output=ptnTrain;
[~,num_frame]=size(signal_train{1});
binary=zeros(nAfferents,num_frame);
for i=1:1:nAfferents
    temp=int16(output{i}*1000);
    [~,num]=size(temp);
    if num>0
       binary(i,temp)=1;
    end  
end



end