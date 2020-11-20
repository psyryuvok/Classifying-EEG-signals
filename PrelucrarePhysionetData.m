%Cod pentru prelucrarea semnalelor de pe physionet
%Semnalele sunt deschise ?i analizate pe rand
 
myFolder = 'C:\Users\Z\Downloads\baza de date\bci\phisyionet\test\';
saveFolder='C:\Users\Z\Downloads\baza de date\bci\phisyionet\test\aved\evenlarger';
% Verific?m dac? fi?ierul chiar exist?.Userul e avertizat in caz c? nu exist? fi?ierul.
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(myFolder, 'S*.edf'); 
theFiles = dir(filePattern);
 
z=0; 
TL1=1;
TL2=1;
TL0=1;
reset=0;
%Se citesc semnalele
for i = 1 : length(theFiles)
  baseFileName = theFiles(i).name;
 
  fprintf(1, 'Now reading %s\n', baseFileName);
 
[hdr, record] = edfread([myFolder,baseFileName]);
[Task_label,Time_duration,Task_sym,strArray] =Eventread(myFolder,baseFileName);
 
 
s=record;
Fs = hdr.frequency(1);                    % Sampling frequency
T = 1/Fs;                     % Sample time
L = length(s);                     % Length of signal
t = (0:L-1)*T;                % Time vector
 
%Eliminarea frecventelor nedorie
s=godfiltc(t,s);
% Verificare filtru
% NFFT=2^nextpow2(L);
% Y=fft(s,NFFT)/L;
% f=Fs/2*linspace(0,1,NFFT/2+1);
% Figura;
% plot(f,2*abs(Y(1:NFFT/2+1)))
 
 
 
% RH(RightHand), LH(LeftHand), RHR(RightHandReal), LHR(LeftHandReal), BH(BothHands), BF(BothFeets), BHR(BothHandsReal),BFR(BothFeetsReal), R1(BreakFromTask1), R2(BreakFromTask2), R3(BreakFromTask3), R4(BreakFromTask4) ,REO(RelaxedEyesOpen) , REC(RelaxedEyesClosed)
% Task 1 (deschide ?i închide pumnul drept sau stâng)
% Task 2 (î?i imagineaz? ca închide ?i deschide pumnul drept sau stâng)
% Task 3 (închide ?i deschide ambii pumni sau picioarele)
% Task 4 (î?i imagineaz? ca închide ?i deschide ambii pumni sau picioarele)
T1=0;
%Se separ? diferitele tipuri de actiuni
num = str2num(baseFileName(end-5:end-3));
if num==3 || num==7 || num==11
 
for i=1:size(Time_duration,1)
if Task_label(i)==1        
        T0=T1+1;
        T1=T0+Time_duration(i)*Fs-1;
        LHR(TL1,:,:)=s(1:64,T0:T0+639);
        TL1=TL1+1;
elseif Task_label(i)==2
    T0=T1+1;
    T1=T0+Time_duration(i)*Fs-1;
    RHR(TL2,:,:)=s(1:64,T0:T0+639);
    TL2=TL2+1;
else
    T0=T1+1;
    T1=T0+Time_duration(i)*Fs-1;
    R1(TL0,:,:)=s(1:64,T0:T0+639);
    TL0=TL0+1;
end
end
 
elseif num==4 || num==8 || num==12
        for i=1:size(Time_duration,1)
if Task_label(i)==1        
        T0=T1+1;
        T1=T0+Time_duration(i)*Fs-1;
        LH(TL1,:,:)=s(1:64,T0:T0+639);
        TL1=TL1+1;
elseif Task_label(i)==2
    T0=T1+1;
    T1=T0+Time_duration(i)*Fs-1;
    RH(TL2,:,:)=s(1:64,T0:T0+639);
    TL2=TL2+1;
else
    T0=T1+1;
    T1=T0+Time_duration(i)*Fs-1;
    R2(TL0,:,:)=s(1:64,T0:T0+639);
    TL0=TL0+1;
end
end
elseif num==5 || num==9 || num==13
            for i=1:size(Time_duration,1)
if Task_label(i)==1        
        T0=T1+1;
        T1=T0+Time_duration(i)*Fs-1;
        BHR(TL1,:,:)=s(1:64,T0:T0+639);
        TL1=TL1+1;
elseif Task_label(i)==2
    T0=T1+1;
    T1=T0+Time_duration(i)*Fs-1;
    BFR(TL2,:,:)=s(1:64,T0:T0+639);
    TL2=TL2+1;
else
    T0=T1+1;
    T1=T0+Time_duration(i)*Fs-1;
    R3(TL0,:,:)=s(1:64,T0:T0+639);
    TL0=TL0+1;
end
end
elseif num==6 || num==10 || num==14
            for i=1:size(Time_duration,1)
if Task_label(i)==1        
        T0=T1+1;
        T1=T0+Time_duration(i)*Fs-1;
        BH(TL1,:,:)=s(1:64,T0:T0+639);
        TL1=TL1+1;
elseif Task_label(i)==2
    T0=T1+1;
    T1=T0+Time_duration(i)*Fs-1;
    BF(TL2,:,:)=s(1:64,T0:T0+639);
    TL2=TL2+1;
else
    T0=T1+1;
    T1=T0+Time_duration(i)*Fs-1;
    R4(TL0,:,:)=s(1:64,T0:T0+639);
    TL0=TL0+1;
end
end
elseif num==2
 
    REC(TL0,:,:)=s(1:64,1:9600);
    TL0=TL0+1;
elseif num==1
    REO(TL0,:,:)=s(1:64,1:9600);
    TL0=TL0+1 ;
else
    z=z+1;
end

%Când se ajunge la valoarea 14, inseamna ca au fost analizate toate inregistr?rile pentru un pacient. Acestea se vor salva apoi ?terge pentru a elibera din RAM
reset=reset+1
if reset==14
%Valorile sunt salvate intr-o celul? ?i apoi într-o structur?
x={RH,LH,RHR,LHR,BH,BF,BHR,BFR,R1,R2,R3,R4,REO,REC};
for k=1:14
type={'RH','LH','RHR','LHR','BH','BF','BHR','BFR','R1','R2','R3','R4','REO','REC'};
mystrucname = sprintf('%s', type{1,k});
xstruc(k).name=mystrucname;
x{1,k}(all(all(x{1,k} ==0,3),2),:,:) = [];
xstruc(k).value=x{1,k}; 
 end
 
 
type=baseFileName(1:end-4);
matfile = fullfile(saveFolder,type);
%Sunt salvate variabilele care con?in valorile mi?c?rilor
save(matfile,'RH','LH','RHR','LHR','BH','BF','BHR','BFR','R1','R2','R3','R4','REO','REC')
clear all
%Sunt ini?ializate variabilele de început
z=0; 
TL1=1;
TL2=1;
TL0=1;
reset=0;
myFolder = 'C:\Users\Z\Downloads\baza de date\bci\phisyionet\test\';
saveFolder='C:\Users\Z\Downloads\baza de date\bci\phisyionet\test\aved\evenlarger';
% Verific?m dac? fi?ierul chiar exist?.Userul e avertizat in caz c? nu exist? fi?ierul.
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(myFolder, 'S*.edf'); 
theFiles = dir(filePattern);
 
end
end
end
