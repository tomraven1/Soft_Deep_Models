siz=50;

% layers = [sequenceInputLayer([128 128 3],'Name','input')
%      sequenceFoldingLayer('Name','fold')
%     convolution2dLayer(5,1,'Padding','same','WeightsInitializer','he','BiasInitializer','zeros','Name','conv');
%     reluLayer('Name','relu')
%     sequenceUnfoldingLayer('Name','unfold')
%     flattenLayer('Name','flatten')
%     dropoutLayer(0.05,'Name','drop')
%     lstmLayer(60,'OutputMode','last','Name','lstm')
%     fullyConnectedLayer(siz,'Name','fc2')
%         regressionLayer("Name","regressionoutput")
%     ];


layers = [sequenceInputLayer([128 128 3],'Name','input')
    sequenceFoldingLayer('Name','fold')
    convolution2dLayer(2,16,'Padding','same','WeightsInitializer','he','BiasInitializer','zeros','Name','conv');
    reluLayer('Name','relu')
    convolution2dLayer(3,8,'Padding','same','WeightsInitializer','he','BiasInitializer','zeros','Name','conv2');
    reluLayer('Name','relu2')
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    dropoutLayer(0.2,'Name','drop')
    lstmLayer(60,'OutputMode','last','Name','lstm')
    fullyConnectedLayer(siz,'Name','fc2')
    regressionLayer("Name","regressionoutput")
    ];

lgraph = layerGraph(layers);
miniBatchSize  = 512*1;

lgraph = connectLayers(lgraph,'fold/miniBatchSize','unfold/miniBatchSize');


input =fileDatastore(fullfile('inp_man'),'ReadFcn',@load,'FileExtensions','.mat');

%input =fileDatastore(fullfile('inp_man'),'ReadFcn',@load,'FileExtensions','.mat');
output=fileDatastore(fullfile('tar_features'),'ReadFcn',@load,'FileExtensions','.mat');


inputDatat = transform(input,@(data) rearrange_datastore_input(data));
outputDatat = transform(output,@(data) rearrange_datastore_output(data));


trainData=combine(inputDatat,outputDatat);



options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',60000, ...
    'InitialLearnRate',0.3*1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',1, ...
    'LearnRateDropPeriod',10000, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','gpu',...
    'Verbose',true)





net=trainNetwork(trainData, lgraph, options);
%net=trainNetwork(trainData,layerGraph(net),options);

% %  asd=read (input);
output2=fileDatastore(fullfile('tar_man'),'ReadFcn',@load,'FileExtensions','.mat');

reset(input)
reset(output)
reset(output2)


[asd,info]=read (input);
[asd2,info2]=read (output);
[asd3,info3] =read (output2);

%net = resetState(net);
for i = 1:990
    [net,YPred(:,i)] = predictAndUpdateState(net,asd.XTrain(:,:,:,i),'ExecutionEnvironment','gpu');
end

% layers_1(1,1).Mean=0;
net2 = assembleNetwork(layers_1);
%
Ysc = reshape(YPred,[1,1,siz,900]);
ypred = predict(net2, Ysc );

act = reshape(asd2.act,[1,1,siz,990]);
ypred2 = predict(net2, act );
act1 = activations(net2,Ysc,'Decoder-Stage-2-Conv-3');
act1=reshape(act1,128,128,1,900*64);

%implay(ypred);
%implay(ypred2);


A=[ypred ypred2(:,:,:,1:900) YTrain(:,:,:,1:900)];

v = VideoWriter('inp_object.avi');

open(v)
writeVideo(v,mat2gray(XTrain(:,:,:,:)));
close(v)


%implay(act1(:,:,:,1:64:end));
y_real=asd3.YTrain;
ytest=asd2.act;
%implay(y_real);
%implay(ypred);

function image = rearrange_datastore_input(data)
image = data.XTrain;
image = num2cell(image, 1:3); % Wrap 1x21x1x100 data in 1x1x1x100 cell
image = image(:); % Reshape 1x1x1x100 cell to 1x100 cell
end
function image = rearrange_datastore_output(data)
image=data.act;
image = num2cell(image,1); % Wrap 1x21x1x100 data in 1x1x1x100 cell
image = image(:);
end



for j=1:55


    baseFileName = fullfile('inp_man',[num2str(j),'.mat']);
    load(baseFileName)
    baseFileName = fullfile('tar_man',[num2str(j),'.mat']);
    load(baseFileName)

    for i = 1:990
        [net,YPred(:,i)] = predictAndUpdateState(net,XTrain(:,:,:,i),'ExecutionEnvironment','gpu');
    end

    % layers_1(1,1).Mean=0;
    net2 = assembleNetwork(layers_1);
    %
    Ysc = reshape(YPred,[1,1,50,990]);
    ypred = predict(net2, Ysc );
    err(j)=immse(ypred,YTrain);
end
