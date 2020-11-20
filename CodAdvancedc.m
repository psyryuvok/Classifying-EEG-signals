function [] = codf(TfullFileName,fullFileName,test,saveFolder,baseFileName,beginingS,endingS )

 %Valoarea variabilei test ne va spune dac? lu?m in considerare ?i labelul epocilor care trebuiau prezise în competi?ie
if test==0
mydata =importdata(fullFileName);
 
  mrk=mydata.mrk;
  cnt=mydata.cnt;
  nfo=mydata.nfo;
 
cnt= 0.1*double(cnt);
s=cnt;
 
Fs = nfo.fs; % Sampling frequency
T = 1/Fs;    % Sample time
L = length(cnt);% Length of signal
t = (0:L-1)*T;  % Time vector
 %beginingS reprezint? num?rul de en?antioane de dinaintea imagin?rii motorii de unde se va incepe salvarea unei epoci.
%endingS reprezint? num?rul de en?antioane de dup? imaginarea motorie unde se va sfâr?i epoca salvat?.
inputS=beginingS+endingS;
%Semnal raw
plot(t,s);
title('Raw signal')
ylabel('Amplitude(µV)')
xlabel('Time(s)')
 %Filtrare semnal
s=godfilt(t,s,Fs);
 
%Separare semnale in clase
   for j=1:size(s,2)
 i1=0;
   i2=0;
   for  i=1:length(mrk.pos)
   if mrk.y(i)==1
i1=i1+1;
 
    RH{1,j}(i1,1:inputS)= s((mrk.pos(i)-beginingS:(mrk.pos(i)+endingS-1)),j);
   
   elseif  mrk.y(i)==2
       i2=i2+1;
 
    RF{1,j}(i2,1:inputS) = s((mrk.pos(i)-beginingS:(mrk.pos(i)+endingS-1)),j);
 
   end    
   end
   end
   
  tm=(0:length(RH{1,2})-1)*T;  
    %Exemplu semnal
 for i=1:1:5
Figura
plot(tm,RH{1,2}(i,:));
title('Signal example')
ylabel('Amplitude(µV)')
xlabel('Time(s)')
 end
   
   
   for i=1:1:(size(RH,2))
   RH1{1,i}=reshape(RH{1,i},1,[]); %transform? matricea într-un rând de vector.
   end
   
      for i=1:1:(size(RF,2))
   RF1{1,i}=reshape(RF{1,i},1,[]); %transform? matricea într-un rând de vector.
      end
tm=(0:length(RH{1,2})-1)*T;  
 
 
%Croscorelare mana dreapta
for i=1:1:(size(RH1,2)-1)
RH1c(i,:)= xcorr(RH1{1,1},RH1{1,(i+1)});
end
%Croscorelarea picior drept
 
for i=1:1:(size(RF1,2)-1)
RF1c(i,:)= xcorr(RH1{1,1},RF1{1,i});
end
 
 
 
%Separarea datelor in 3 trialuri
 RS1{1,1}=RF1c(1,:);
i= 2;
while i<=(size(RF1c,1)/3)
    x{1,1}=RF1c(i,:);
  RS1=[RS1;x];
   i=i+ 1;
end
i=1;
while i<=(size(RH1c,1)/3)
     x{1,1}=RH1c(i,:);
  RS1=[RS1;x];
   i=i+ 1;
end
 
RS2{1,1}=RF1c(40,:);
i=round((size(RF1c,1)/3)+2);
while i<=((size(RF1c,1)/3)*2)
  RS2=[RS2;RF1c(i,:)];
   i=i+ 1;
end
i=round(size(RH1c,1)/3+1);
while i<=((size(RH1c,1)/3)*2)
  RS2=[RS2;RH1c(i,:)];
   i=i+ 1;
end
 
 
RS3{1,1}=RF1c(79,:);
i=round((size(RF1c,1)/3)*2+2);
while i<=((size(RF1c,1)))
  RS3=[RS3;RF1c(i,:)];
   i=i+ 1;
end
i=round((size(RH1c,1)/3)*2+1);
while i<=((size(RH1c,1)))
  RS3=[RS3;RH1c(i,:)];
   i=i+ 1;
end
 
RSF=[RS1,RS2,RS3];
 
%Determinarea variabilelor independente
for j=1:1:size(RSF,2)
    for i=1:1:size(RSF,1)
x1(i,j)=mean(RSF{i,j}(1,:));
x2(i,j)=max(RSF{i,j}(1,:));
x3(i,j)=min(RSF{i,j}(1,:));
x4(i,j)=std(RSF{i,j}(1,:));
x5(i,j)=median(RSF{i,j}(1,:));
x6(i,j)=mode(RSF{i,j}(1,:));
    end
end
 

%cell save
x={x1,x2,x3,x4,x5,x6};
%struct save, in caz c? se dore?te s? se lucreze cu structuri.
% 
% strr=struct()
% [strr(:).mean] = deal(x{1,1});
% [strr(:).max] = deal(x{1,2});
% [strr(:).min] = deal(x{1,3});
% [strr(:).std] = deal(x{1,4});
% [strr(:).median] = deal(x{1,5});
% [strr(:).mode] = deal(x{1,6});
 
 
%eliminat split 3
x1=x1(:);
x2=x2(:);
x3=x3(:);
x4=x4(:);
x5=x5(:);
x6=x6(:);
X(:,1)=x1;
X(:,2)=x2;
X(:,3)=x3;
X(:,4)=x4;
X(:,5)=x5;
X(:,6)=x6;
 
type = 'P';
s = strcat(type,baseFileName);
matfile = fullfile(saveFolder, s);
%Salvarea variabilelor de interes
save(matfile,'X','RH','RF')
%In c? se dore?te luarea in considerare a labelurilor suplimentare
elseif test==1
    
mydata =importdata(fullFileName);
true_y=importdata(TfullFileName);
      
  mrk=mydata.mrk;
  cnt=mydata.cnt;
  nfo=mydata.nfo;
  mrk.y=true_y.true_y;
    
 
  
 
cnt= 0.1*double(cnt);
s=cnt;
 
Fs = nfo.fs; % Sampling frequency
T = 1/Fs;    % Sample time
L = length(cnt); % Length of signal
t = (0:L-1)*T;   % Time vector
 
%beginingS reprezint? num?rul de en?antioane de dinaintea imagin?rii motorii de unde se va incepe salvarea unei epoci.
%endingS reprezint? num?rul de en?antioane de dup? imaginarea motorie unde se va sfâr?i epoca salvat?.

inputS=beginingS+endingS;
%Semnal raw
plot(t,s);
title('Raw signal')
ylabel('Amplitude(µV)')
xlabel('Time(s)')
 
s=godfilt(t,s,Fs);
 
%Separare semnale in clase
   for j=1:size(s,2)
 i1=0;
   i2=0;
   for  i=1:length(mrk.pos)
   if mrk.y(i)==1
i1=i1+1;
 
    RH{1,j}(i1,1:inputS)= s((mrk.pos(i)-beginingS:(mrk.pos(i)+endingS-1)),j);
   
   elseif  mrk.y(i)==2
       i2=i2+1;
 
    RF{1,j}(i2,1:inputS) = s((mrk.pos(i)-beginingS:(mrk.pos(i)+endingS-1)),j);
 
   end    
   end
   end
   
  tm=(0:length(RH{1,2})-1)*T;  
    %Exemplu semnal
 for i=1:1:5
Figura
plot(tm,RH{1,2}(i,:));
title('Signal example')
ylabel('Amplitude(µV)')
xlabel('Time(s)')
 end
   
   
   for i=1:1:(size(RH,2))
   RH1{1,i}=reshape(RH{1,i},1,[]); %transform? matricea într-un rând de vector.
   end
   
      for i=1:1:(size(RF,2))
   RF1{1,i}=reshape(RF{1,i},1,[]); %transform? matricea într-un rând de vector.
      end
tm=(0:length(RH{1,2})-1)*T;  
 
 
 
%Croscorelare mana dreapta
for i=1:1:(size(RH1,2)-1)
RH1c(i,:)= xcorr(RH1{1,1},RH1{1,(i+1)});
end
%Croscorelarea picior drept
 
for i=1:1:(size(RF1,2)-1)
RF1c(i,:)= xcorr(RH1{1,1},RF1{1,i});
end
 
 
 
 
%Separarea datelor in 3 trialuri
 RS1{1,1}=RF1c(1,:);
i= 2;
while i<=(size(RF1c,1)/3)
    x{1,1}=RF1c(i,:);
  RS1=[RS1;x];
   i=i+ 1;
end
i=1;
while i<=(size(RH1c,1)/3)
     x{1,1}=RH1c(i,:);
  RS1=[RS1;x];
   i=i+ 1;
end
 
RS2{1,1}=RF1c(40,:);
i=round((size(RF1c,1)/3)+2);
while i<=((size(RF1c,1)/3)*2)
  RS2=[RS2;RF1c(i,:)];
   i=i+ 1;
end
i=round(size(RH1c,1)/3+1);
while i<=((size(RH1c,1)/3)*2)
  RS2=[RS2;RH1c(i,:)];
   i=i+ 1;
end
 
 
RS3{1,1}=RF1c(79,:);
i=round((size(RF1c,1)/3)*2+2);
while i<=((size(RF1c,1)))
  RS3=[RS3;RF1c(i,:)];
   i=i+ 1;
end
i=round((size(RH1c,1)/3)*2+1);
while i<=((size(RH1c,1)))
  RS3=[RS3;RH1c(i,:)];
   i=i+ 1;
end
 
RSF=[RS1,RS2,RS3];
 
%Determinarea variabilelor independente
for j=1:1:size(RSF,2)
    for i=1:1:size(RSF,1)
x1(i,j)=mean(RSF{i,j}(1,:));
x2(i,j)=max(RSF{i,j}(1,:));
x3(i,j)=min(RSF{i,j}(1,:));
x4(i,j)=std(RSF{i,j}(1,:));
x5(i,j)=median(RSF{i,j}(1,:));
x6(i,j)=mode(RSF{i,j}(1,:));
    end
end
 
 
 
%cell save
x={x1,x2,x3,x4,x5,x6};
%struct save, in caz c? se dore?te s? se lucreze cu structuri.
% 
% strr=struct()
% [strr(:).mean] = deal(x{1,1});
% [strr(:).max] = deal(x{1,2});
% [strr(:).min] = deal(x{1,3});
% [strr(:).std] = deal(x{1,4});
% [strr(:).median] = deal(x{1,5});
% [strr(:).mode] = deal(x{1,6});
 
 
%eliminat split 3
x1=x1(:);
x2=x2(:);
x3=x3(:);
x4=x4(:);
x5=x5(:);
x6=x6(:);
X(:,1)=x1;
X(:,2)=x2;
X(:,3)=x3;
X(:,4)=x4;
X(:,5)=x5;
X(:,6)=x6;
 
 
 
type = 'TP';
s = strcat(type,baseFileName);
matfile = fullfile(saveFolder, s);
%Salvarea variabilelor de interes
save(matfile,'X','RH','RF')
 
else
    disp('Trebuie sa alegi 0 sau 1 la test')
    fprintf ( 1, '0 reprez fara test label iar 1 cu' );
end
end
