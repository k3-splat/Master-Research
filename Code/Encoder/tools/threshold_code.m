function [spike_train,T]=threshold_code(input_signal)


N_channel=20;
afferent=N_channel*(2*15+1);
spike_train=cell(afferent,1);


[~, time]=size(input_signal);
T=0.001*(time);    
    
for f=1:1:N_channel
    
    for t=1:1:time-1    
        current=input_signal(f,t);
        current_time=(t+1)*0.001;
        
        if current==1  % special  
            spike_train{31*f}=[spike_train{31*f},current_time];
        else
            fire_index=find_index(input_signal(f,t),input_signal(f,t+1));
            if fire_index==100
                continue
            else
                spike_train{fire_index+31*(f-1)}=[spike_train{fire_index+31*(f-1)},current_time];
            end
        end          
    end    
end    
    
end



