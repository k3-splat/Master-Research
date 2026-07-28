function fire_index=find_index(input_signal1,input_signal2);

% fire_index:  0.0625*[1:1:15,15:-1:1];
onset=0.0625*[1:1:15];
offset=0.0625*[15:-1:1];
cross=input_signal2-input_signal1;

if cross>0
    temp=find(onset>input_signal1);  
    temps=size(temp);
    if  temps(2)==0
        fire_index=100;
    else
        index=temp(1);
        if onset(index)<input_signal2 
            fire_index=index;
        else
           fire_index=100; 
        end     
    end    



else
    temp=find(offset<input_signal1);                   % consider null matrix
    temps=size(temp);
    if  temps(2)==0
        fire_index=100;
    else
        index=temp(1);
        if offset(index)>input_signal2 
            fire_index=index+15;
        else
           fire_index=100; 
        end     
    end
    
end
        
      












end