function  [masker]=mask_generate(signal,window_length,x,cf)


channel=20;
Ls=length(signal);  % 20kHz sampling rate
step=window_length/2;
loops=floor(Ls/step); 
energy_spec=zeros(channel,loops);
fre_energy=zeros(channel,loops);
masking=zeros(channel,loops);

    temp_signal=signal;
    temp_signal=[temp_signal(1:loops*step);zeros(step,1)];
    for i=1:1:loops
        temp=temp_signal((i-1)*step+1:(i-1)*step+window_length);
        
        for bin=1:1:channel
            
            f=abs(fft(temp,16000));
            f=f(201:5002);
            z=x(bin,:).*f';  %x is the filterbank with 20 channels
            fre_energy(bin,i)=sum(z.^2);
            
            energy_spec(bin,i)=10*log10(sum(z.^2));
        end
        
        cbpower=energy_spec(:,i);
        fc=cf;
        masking(:,i) = cbthr1(cbpower,fc);        
    end

   
energy_spec=energy_spec-min(min(energy_spec));
masker=ones(channel,loops);
masker(energy_spec<masking)=0;

end