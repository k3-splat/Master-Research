function mpks = my_mask_modified(pks,th_init)

%tau=0.008:0.0011:0.0289;
%c=exp(-tau);

%c=[0.80:-0.001:0.80-0.019]; %standard parmeter for experiment results

c=[0.93:-0.001:0.93-0.019];

%c=[0.93-0.019:0.001:0.93];
%c=[0.97:-0.001:0.97-0.019];

threshold=0.0625;

[filters,L] = size(pks);
mpks = ones(size(pks));
%Masked_ind=zeros(size(pks));

for k = 1:filters,

 
   thr(1) = th_init(k);
   for l = 2:L,
      if pks(k,l)>c(k)*thr(l-1),  %different filter has different decaying speed c
         thr(l) = pks(k,l);

      else
         thr(l) = c(k)*thr(l-1);
      end
      
      %% logical module:
      if pks(k,l) < thr(l)  
        if pks(k,l-1)-pks(k,l)>threshold*1.5 
            mpks(k,l)=1;
        else
            mpks(k,l)=0;
        end
      else
          mpks(k,l)=1;
      end
      
      
   end
   
   

   
end