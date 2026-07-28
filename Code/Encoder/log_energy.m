function cochlea_spectro=log_energy(parallel_stream,window_len)

eps=10^(-10);
[channel,length]=size(parallel_stream);
step=window_len/2; 
loops=ceil(length/step)-1; 
cochlea_spectro=zeros(channel,loops);


for j=1:1:channel
    temp_signal=parallel_stream(j,:)';
    temp_signal=[temp_signal;zeros((window_len-step-rem(length,step)),1)];
        for i=1:1:loops
            temp=temp_signal((i-1)*step+1:(i-1)*step+window_len);
            S=sum(temp.*temp);
            cochlea_spectro(j,i)=log(S+eps)-log(eps);    
        end    
    
end


end







