%Cod pentru apelarea func?iei codf ?i preg?tire date pentru Python
prompt = 'Luam in considerare test_labels? 1=da 0=nu ';
test = input(prompt)
myFolder = 'C:\Users\Z\Downloads\baza de date\bci\testing';
saveFolder='C:\Users\Z\Downloads\baza de date\bci\testing\processed\checking\New folder\corectformat';
%Verific?m dac? fi?ierul chiar exist?.Userul e avertizat in caz c? nu exist? fi?ierul.

if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(myFolder, 'data_set*.mat'); % Change to whatever pattern you need.
TfilePattern = fullfile(myFolder, 'true_labels*.mat');  
theFiles = dir(filePattern);
TtheFiles = dir(TfilePattern);
for jkl = 1 : length(theFiles)
  baseFileName = theFiles(jkl).name;
  fullFileName = fullfile(myFolder, baseFileName);
  TbaseFileName = TtheFiles(jkl).name;
  TfullFileName = fullfile(myFolder, TbaseFileName);
 
  fprintf(1, 'Now reading %s\n', fullFileName);
  CodAdvancedc( TfullFileName,fullFileName,test,saveFolder,baseFileName,100,400);
End
%Se vor închide toate plot-urile deschise
close all
%pregatire variabilei care con?ine label
y=reshape([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[78,1]);
y=[y; y;y];
%Pregatit date pentru a putea fi procesate in python cu cele 2 metode.
 
if test==0
    RH={1};
    RF={1};
for jkl=1:length(theFiles)
    
  baseFileName = theFiles(jkl).name;    
  type = 'P';
  s = strcat(type,baseFileName);
  fullFileName = fullfile(saveFolder, s);
  
d= importdata(fullFileName);
X(:,:,jkl)=d.X;
RH={RH{1,:},d.RH{1,:}};
RF={RF{1,:},d.RF{1,:}};
end
prompt = 'Name of saves? ';
type= input(prompt,'s');
matfile = fullfile(saveFolder,type);
save(matfile,'X','RF','RH','y')
 
 
 
elseif test==1
    RH={1};
    RF={1};
    for jkl=1:length(theFiles)
  baseFileName = theFiles(jkl).name;    
  type = 'TP';
  s = strcat(type,baseFileName);
  fullFileName = fullfile(saveFolder, s);
  
d= importdata(fullFileName);
X(:,:,jkl)=d.X;
RH={RH{1,:},d.RH{1,:}};
RF={RF{1,:},d.RF{1,:}};
end
prompt = 'Name of Tsaves? ';
type= input(prompt,'s');
matfile = fullfile(saveFolder,type);
save(matfile,'X','RF','RH','y')
 
 
else
    disp('Trebuie sa alegi 0 sau 1 la test')
    fprintf ( 1, '0 reprez fara test label iar 1 cu' );
end
clear all
