function average_result=average_withMasking(signal,window_length)

[channel,length]=size(signal);
%win_length=800; % 50ms



step=window_length/2; % 25ms
%step=1;

%window=hamming(window_length);


%loops=floor(length/step);  % for original

loops=ceil(length/step)-1; 



average_result=zeros(channel,loops);

%pad=loops*step-length(signal);

for j=1:1:channel
    temp_signal=signal(j,:)';
    temp_signal=[temp_signal;zeros((window_length-step-rem(length,step)),1)];
    for i=1:1:loops
        temp=temp_signal((i-1)*step+1:(i-1)*step+window_length);
        

        %average_result(j,i)=log(sum(temp.*temp));  % get the log energy within the window 
        S=sum(temp.*temp);  % IF USING TEMPORAL MASKING, TOTAL ENERGY MAY NOT WORK How to framing?
        eps=10^(-10);
        average_result(j,i)=log(S+eps)-log(eps);    
    end    
    
end


end