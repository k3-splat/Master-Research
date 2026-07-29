function output=global_normalization(input_cellarray)

%% Method 1
% [N,~]=size(input_cellarray);
% 
% output=cell(N,1);
% 
% max_val=zeros(1,N);
% min_val=zeros(1,N);
% 
% for i=1:1:N
%     
%     max_val(i)=max(max(input_cellarray{i}));
%     min_val(i)=min(min(input_cellarray{i}));
% 
% end
% 
% global_max=max(max_val);
% global_min=min(min_val);
% 
% 
% for i=1:1:N
%     
%     sample=input_cellarray{i};
%     
%     output{i}=(sample-global_min)./(global_max-global_min);
% 
% end


% [N,~]=size(input_cellarray);
% 
% output=cell(N,1);
% 
% max_val=zeros(1,N);
% min_val=zeros(1,N);

%% Method 2

%  [N,~]=size(input_cellarray);
%  
% output=cell(N,1);
% 
% 
% % max_temp=zeros(20,N);
% % min_temp=zeros(20,N);
% 
% % for i=1:1:N
% %     
% %     
% %     tempMatrix=input_cellarray{i};
% %     max_temp(:,i)=max(tempMatrix,[],2);
% %     min_temp(:,i)=min(tempMatrix,[],2);
% %     
% % end
% % 
% % max_val=max(max_temp,[],2);
% % min_val=min(min_temp,[],2);
% 
% 
% for i=1:1:N
%     
%     for f=1:1:20
%         temp=input_cellarray{i}; 
%         
%         max_val=max(temp,[],2);
%         min_val=min(temp,[],2);        
%         
%         output{i}=[output{i};(temp(f,:)-min_val(f))./(max_val(f)-min_val(f))];
% 
%     end
%     
% end

%% Method 3

% Only normalize within one sample. Dont bother with any other samples

%  [N,~]=size(input_cellarray);
%  
% output=cell(N,1);
% for i=1:1:N
%     
%     for f=1:1:20
%         sample=input_cellarray{i};  
%         max_val=max(sample,[],2);
%         min_val=min(sample,[],2);
%         
%         max_global=max(max_val);
%         min_global=min(min_val);
%         
%         temp_sample=(sample-min_global)/(max_global-min_global);
%         
%         eps=10^(-5);
%         temp=log(temp_sample+eps)-log(eps);
%         
%         
%         
%         
%         max_val=max(temp,[],2);
%         min_val=min(temp,[],2);        
%         output{i}=[output{i};(temp(f,:)-min_val(f))./(max_val(f)-min_val(f))];
% 
%     end
%     
% end

%% Method 4: global min-max within each training data
 [N,~]=size(input_cellarray);
N_channel=20;
output=cell(N,1);


for i=1:1:N
    
    for f=1:1:N_channel
        temp=input_cellarray{i}; 
        
        max_val=max(max(temp));
        min_val=min(min(temp));        
        
        output{i}=[output{i};(temp(f,:)-min_val)/(max_val-min_val)];

    end
    
end







end