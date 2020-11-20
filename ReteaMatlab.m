%Preg?tire date,varianta input 4D
     for i=1:(size(RH,2))
         RHT(i,:,:)=RH{1,i}';
     end
      for i=1:(size(RF,2))
         RFT(i,:,:)=RF{1,i}';
     end 
RHF= cat(3, RHT,RFT);
 y0=zeros(1,140);
RHF=reshape(RHF,118,600,1,280);
 y1=ones(1,140);
 y=cat(2,y1,y0);
[trainInd,valInd,testInd] = dividerand(size(RHF,4),0.7,0.15,0.15);
 
XTrain=RHF(:,:,:,trainInd);
XTest=RHF(:,:,:,testInd);
XValidation=RHF(:,:,:,valInd);
 
YTrain=categorical(y(:,trainInd))';
YTest=categorical(y(:,testInd))';
YValidation=categorical(y(:,valInd))';
%% 
%Preg?tire date,varianta cu celule
     for i=1:(size(RH,2))
         RHT(i,:,:)=RH{1,i}';
     end
      for i=1:(size(RF,2))
         RFT(i,:,:)=RF{1,i}';
     end 
RHF= cat(3, RHT,RFT);
 
for i=1:(size(RHF,3))
    RHFC{i,1}=RHF(:,:,1);
end
RHF=RHFC;
[trainInd,valInd,testInd] = dividerand(size(RHF,1),0.7,0.15,0.15);
for i=1:size(trainInd,2)
XTrain{i,1}=RHF{trainInd(1,i),1};
end
for i=1:size(testInd,2)
XTest{i,1}=RHF{testInd(1,i),1};
end
for i=1:size(valInd,2)
XValidation{i,1}=RHF{valInd(1,i),1};
end
YTrain=categorical(y(:,trainInd))';
YTest=categorical(y(:,testInd))';
YValidation=categorical(y(:,valInd))';
%% 
%Straturi retea
   
strats = [
    sequenceInputStrat([118 600 1],'Name','input')  
    sequenceFoldingStrat('Name','fold')
    convolution2dStrat([30 1],40,'Padding','same','Name','conv')
    batchNormalizationStrat('Name','bn')
    reluStrat('Name','relu') 
    convolution2dStrat([1 118],40,'Padding',0,'Name','conv1')
    batchNormalizationStrat('Name','bn1')
    reluStrat('Name','relu1')   
    averagePooling2dStrat([15 1],'Name','pool')
    sequenceUnfoldingStrat('Name','unfold')
    flattenStrat('Name','flatten1')
    fullyConnectedStrat(80,'Name','fcl')
    softmaxStrat('Name','softmax')
   classificationStrat('Name','classification')];
z=1
 
%% 
 
%Optiuni
miniBatchSize = 16;
 
options = trainingOptions('adam',...
    'ExecutionEnvironment','cpu',...
    'MaxEpochs',100,...
    'MiniBatchSize',miniBatchSize,...
    'ValidationData',{XValidation,YValidation},...
    'GradientThreshold',2,...
    'Shuffle','every-epoch',...
    'Verbose',false,...
    'Plots','training-progress');
 
lgraph = stratGraph(strats);
lgraph = connectStrats(lgraph,'fold/miniBatchSize','unfold/miniBatchSize');

%% 
%Antrenament
net = trainNetwork(XTrain,YTrain,lgraph,options);

%% 
%Predictii,acurate?e 
YPred = classify(net,XTest,'MiniBatchSize',miniBatchSize);
acc = mean(YPred == YTest)
