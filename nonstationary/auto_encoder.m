% YTrain=single(BW);
% %YTrain=reshape(YTrain,[128,128,3,20000]);
% YTrain=YTrain/255;
% YTrain=YTrain(:,:,:,1:1000);


layers = [
    imageInputLayer([128,128,3],'Name','input_encoder','Normalization','none')
    convolution2dLayer(3, 32*2, 'Padding','same', 'Stride', 2, 'Name', 'conv1')
    reluLayer('Name','relu1')
    convolution2dLayer(3, 64*2, 'Padding','same', 'Stride', 2, 'Name', 'conv2')
    reluLayer('Name','relu2')
       convolution2dLayer([3 3] , 64*2, 'Padding','same', 'Stride', 2, 'Name', 'co545v2')
    reluLayer('Name','re5452')
           convolution2dLayer([2 2] , 32*2, 'Padding','same', 'Stride', 2, 'Name', 'c45v2')
    reluLayer('Name','re82')
    dropoutLayer(0.02,'Name','drop1')
    fullyConnectedLayer(50, 'Name', 'fc_encoder')
    reluLayer('Name','reldd')
    transposedConv2dLayer(7, 64, 'Cropping', 'same', 'Stride', 7, 'Name', 'transpose1')
    reluLayer('Name','relue')
    transposedConv2dLayer(3, 64, 'Cropping', 'same', 'Stride', 2, 'Name', 'transpose2')
    reluLayer('Name','rele')
    transposedConv2dLayer(3, 64, 'Cropping', 'same', 'Stride', 2, 'Name', 'transpose3')
    reluLayer('Name','reweu3')
    transposedConv2dLayer(3, 64, 'Cropping', 'same', 'Name', 'transpose4')
    resize2dLayer("Name","resize-output-size2","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[128 128])
    convolution2dLayer([3 3],64,"Name","Decoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-ReLU-1")
    convolution2dLayer([3 3],128,"Name","Decoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-ReLU-2")
    convolution2dLayer([2 2],128,"Name","Decoder34-45e-1-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-S-1-Re34")
    convolution2dLayer([2 2],64,"Name","Decoder-Stage-2-Conv-3","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-ReLU-3")
    convolution2dLayer([1 1],3,"Name","Final-ConvolutionLayer","Padding","same","WeightsInitializer","he")
    
    regressionLayer("Name","regressionoutput")
    ];

lgraph = layerGraph(layers);

input =fileDatastore(fullfile('tar_man'),'ReadFcn',@load,'FileExtensions','.mat');

%input =fileDatastore(fullfile('inp_man'),'ReadFcn',@load,'FileExtensions','.mat');
output=fileDatastore(fullfile('tar_man'),'ReadFcn',@load,'FileExtensions','.mat');


inputDatat = transform(input,@(data) rearrange_datastore_input(data));
outputDatat = transform(output,@(data) rearrange_datastore_output(data));


trainData=combine(outputDatat,outputDatat);


input2 =fileDatastore(fullfile('tar_man_val'),'ReadFcn',@load,'FileExtensions','.mat');
output2=fileDatastore(fullfile('tar_man_val'),'ReadFcn',@load,'FileExtensions','.mat');
inputDatat2 = transform(input,@(data) rearrange_datastore_input(data));
outputDatat2 = transform(output,@(data) rearrange_datastore_output(data));


valData=combine(outputDatat2,outputDatat2);

miniBatchSize  = 64;

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',60000, ...
    'InitialLearnRate',0.3*1e-3, ...
    'ValidationData',valData, ...
    'ValidationFrequency',3000, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',1, ...
    'LearnRateDropPeriod',10000, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','gpu',...
    'Verbose',true)





net=trainNetwork(trainData, lgraph, options);
%net=trainNetwork(trainData,layerGraph(net),options);

  asd=read (input);
% asd=read (input);
 ypred = predict(net,asd.YTrain(:,:,:,1:900)); 

  asd=read (input);
 act1 = activations(net,asd.YTrain(:,:,:,1:900),'Decoder-Stage-2-Conv-3');
 reshape(act1,128,128,1,900*64);

function image = rearrange_datastore_input(data)
image = data.XTrain;
image = num2cell(image, 1:3); % Wrap 1x21x1x100 data in 1x1x1x100 cell
image = image(:); % Reshape 1x1x1x100 cell to 1x100 cell
end
function image = rearrange_datastore_output(data)
image=data.YTrain;
image = num2cell(image, 1:3); % Wrap 1x21x1x100 data in 1x1x1x100 cell
image = image(:);
end




for i=1:55
    baseFileName = fullfile('tar_man',[num2str(i),'.mat']);
    load(baseFileName)
    act = activations(net,YTrain(:,:,:,:),'fc_encoder');
    act=squeeze(act);
    filename=sprintf('tar_features/%d.mat',i);
    save(filename,'act')
end





%net=trainNetwork(YTrain,YTrain, lgraph, options);
%net=trainNetwork(YTrain,YTrain,layerGraph(net),options);
%ypred = predict(net,YTrain);


%act = activations(net,YTrain,'fc_encoder');


% num2cell(BW,[1 2 3]);
% squeeze(ans);
%
% autoenc = trainAutoencoder(ans,50);
%
%
% xReconstructed = predict(autoenc,ans);

% latentDim = 20;
% imageSize = [128 128 3];
% lgraph = layerGraph();
%
% encoderLG = layerGraph([
%     imageInputLayer(imageSize,'Name','input_encoder','Normalization','none')
%     convolution2dLayer(3, 32, 'Padding','same', 'Stride', 2, 'Name', 'conv1')
%     reluLayer('Name','relu1')
%     convolution2dLayer(3, 64, 'Padding','same', 'Stride', 2, 'Name', 'conv2')
%     reluLayer('Name','relu2')
%     fullyConnectedLayer(2 * latentDim, 'Name', 'fc_encoder')
%     ]);
%
% lgraph = addLayers(lgraph,encoderLG);
%
% tempLayers= layerGraph([
%     imageInputLayer([1 1 latentDim],'Name','i','Normalization','none')
%     transposedConv2dLayer(7, 64, 'Cropping', 'same', 'Stride', 7, 'Name', 'transpose1')
%     reluLayer('Name','relu1')
%     transposedConv2dLayer(3, 64, 'Cropping', 'same', 'Stride', 2, 'Name', 'transpose2')
%     reluLayer('Name','relu2')
%     transposedConv2dLayer(3, 32, 'Cropping', 'same', 'Stride', 2, 'Name', 'transpose3')
%     reluLayer('Name','relu3')
%     transposedConv2dLayer(3, 1, 'Cropping', 'same', 'Name', 'transpose4')
%     ]);
%
% lgraph = addLayers(lgraph,tempLayers);
%
% clear tempLayers;
% lgraph = connectLayers(lgraph,"input_encoder","i");