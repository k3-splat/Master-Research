 %% encode the TIMIT dataset
clear all;clc;
 dbstop if error;

 train_phone_label=cell(2,1);
 train_text_label=cell(2,1);
 train_word_label=cell(2,1);
 train_TIMIT_data=cell(2,1);
 train_word_data=cell(2,1);
  
addpath('~\tools');

train_dir='F:\TIMIT masking\timit\train';
test_dir='F:\TIMIT masking\timit\test';
train_sub1=dir(train_dir);
test_sub1=dir(test_dir);

counter=1;
for i=3:length(train_sub1)
    
    temp2=[train_dir,'\',train_sub1(i).name];
    train_sub2=dir(temp2);
    
    for j=3:length(train_sub2)
        
        temp3=[train_sub2(j).folder,'\',train_sub2(j).name];
        train_sub3=dir(temp3);
         
        for k=3:4:length(train_sub3)
            
                
                %phn_file=train_sub3(k).name;
                %txt_file=train_sub3(k+1).name;
                [wav_file,fs]=audioread([train_sub3(k+2).folder,'\',train_sub3(k+2).name]);

               
                [spike_output,binary_output]=encode_single_wav(wav_file); 
                
                file_name=[train_sub3(k+2).folder,'\',train_sub3(k+2).name];
                name1=[file_name(1:end-4),'_spike_timing','.mat'];
                name2=[file_name(1:end-4),'_binary_matrix','.mat'];
                save(name1,'spike_output');
                save(name2,'binary_output');              
                counter=counter+1
        end
        
    end
    
end
        
%save('train_TIMIT_data.mat','train_TIMIT_data');
    

        
%% testing data

 test_phone_label=cell(2,1);
 test_text_label=cell(2,1);
 test_word_label=cell(2,1);
 test_TIMIT_data=cell(2,1);
 test_word_data=cell(2,1);
  
addpath('G:\Resarch Work\Threshold code and temporal masking');
addpath('F:\TIMIT\timit');

test_dir='F:\TIMIT\timit\test';
test_sub1=dir(test_dir);

counter=1;
except=0;
for i=3:length(test_sub1)
    
    temp2=[test_dir,'\',test_sub1(i).name];
    test_sub2=dir(temp2);
    
    for j=3:length(test_sub2)
        
        temp3=[test_sub2(j).folder,'\',test_sub2(j).name];
        test_sub3=dir(temp3);
         
        for k=3:4:length(test_sub3)
            
            try
                %phn_file=train_sub3(k).name;
                %txt_file=train_sub3(k+1).name;
                [wav_file,fs]=audioread([test_sub3(k+2).folder,'\',test_sub3(k+2).name]);
 
                
                [spike_output,binary_output]=encode_single_wav(wav_file); 
                
                file_name=[test_sub3(k+2).folder,'\',test_sub3(k+2).name];
                name1=[file_name(1:end-4),'_spike_timing','.mat'];
                name2=[file_name(1:end-4),'_binary_matrix','.mat'];
                save(name1,'spike_output');
                save(name2,'binary_output');  
                
                counter=counter+1
                
            catch exception
                   fid=fopen([test_sub3(k+1).folder,'\',test_sub3(k+1).name]);
                   %test_TIMIT_data{counter,1}=fread(fid);
                   
                   wav_file=fread(fid);
                   
                    
                    [spike_output,binary_output]=encode_single_wav(wav_file); 
                    file_name=[test_sub3(k+2).folder,'\',test_sub3(k+2).name];
                    name1=[file_name(1:end-4),'_spike_timing','.mat'];
                    name2=[file_name(1:end-4),'_binary_matrix','.mat'];
                    save(name1,'spike_output');
                    save(name2,'binary_output');  
                   
                   counter=counter+1
                   except=except+1;
            end
                    
        end
        
    end
    
end
%save('test_TIMIT_data.mat','test_TIMIT_data');


