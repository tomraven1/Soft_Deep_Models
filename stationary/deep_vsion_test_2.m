% lgraph = layerGraph();
%
%
% tempLayers = [
%     imageInputLayer([128 128 3],"Name","ImageInputLayer",'Normalization','none')%imageInputLayer([32 32 1],"Name","ImageInputLayer",'Normalization','none')
%     convolution2dLayer([3 3],16,"Name","Encoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Encoder-Stage-1-ReLU-1")
%     convolution2dLayer([3 3],16,"Name","Encoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Encoder-Stage-1-ReLU-2")];
% lgraph = addLayers(lgraph,tempLayers);
%
% tempLayers = [
%     maxPooling2dLayer([2 2],"Name","Encoder-Stage-1-MaxPool","Stride",[2 2])
%     convolution2dLayer([3 3],64,"Name","Encoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Encoder-Stage-2-ReLU-1")
%     convolution2dLayer([3 3],64,"Name","Encoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Encoder-Stage-2-ReLU-2")];
% lgraph = addLayers(lgraph,tempLayers);
%
% tempLayers = [
%     maxPooling2dLayer([2 2],"Name","Encoder-Stage-2-DropOut","Stride",[2 2])
%     convolution2dLayer([3 3],256,"Name","Bridge-Conv-1","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Bridge-ReLU-1")
%     convolution2dLayer([3 3],256,"Name","Bridge-Conv-2","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Bridge-ReLU-2")
%     dropoutLayer(0.1,"Name","Bridge-DropOut")
%     transposedConv2dLayer([2 2],128,"Name","Decoder-Stage-1-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
%     reluLayer("Name","Decoder-Stage-1-UpReLU")];
% lgraph = addLayers(lgraph,tempLayers);
%
% tempLayers = [
%     depthConcatenationLayer(2,"Name","Decoder-Stage-1-DepthConcatenation")
%     convolution2dLayer([3 3],128,"Name","Decoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Decoder-Stage-1-ReLU-1")
%     convolution2dLayer([3 3],128,"Name","Decoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Decoder-Stage-1-ReLU-2")
%     transposedConv2dLayer([2 2],128,"Name","Decoder-Stage-2-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
%     reluLayer("Name","Decoder-Stage-2-UpReLU")];
% lgraph = addLayers(lgraph,tempLayers);
%
% tempLayers = [
%     depthConcatenationLayer(2,"Name","Decoder-Stage-2-DepthConcatenation")
%     resize2dLayer("Name","resize-output-size2","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[128 128])
%     convolution2dLayer([3 3],64,"Name","Decoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Decoder-Stage-2-ReLU-1")
%     convolution2dLayer([3 3],64,"Name","Decoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
%         reluLayer("Name","Decoder-Stage-2-ReLU-3")
%     convolution2dLayer([3 3],64,"Name","Decoder-Stage-2-Conv-3","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Decoder-Stage-2-ReLU-2")
%     convolution2dLayer([1 1],3,"Name","Final-ConvolutionLayer","Padding","same","WeightsInitializer","he")
%     % resize2dLayer("Name","resize-output-size","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[128 128])
%     regressionLayer("Name","regressionoutput")];
% lgraph = addLayers(lgraph,tempLayers);
% %    resize2dLayer("Name","resize-output-size","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[128 128]
% % clean up helper variable
% clear tempLayers;
%
% lgraph = connectLayers(lgraph,"Encoder-Stage-1-ReLU-2","Encoder-Stage-1-MaxPool");
% lgraph = connectLayers(lgraph,"Encoder-Stage-1-ReLU-2","Decoder-Stage-2-DepthConcatenation/in2");
% lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","Encoder-Stage-2-DropOut");
% lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","Decoder-Stage-1-DepthConcatenation/in2");
% lgraph = connectLayers(lgraph,"Decoder-Stage-1-UpReLU","Decoder-Stage-1-DepthConcatenation/in1");
% lgraph = connectLayers(lgraph,"Decoder-Stage-2-UpReLU","Decoder-Stage-2-DepthConcatenation/in1");

lgraph = layerGraph();

tempLayers = [
    imageInputLayer([32 32 1],'Name','input_encoder','Normalization','none')
    convolution2dLayer(3, 64, 'Padding','same', 'Stride', 1, 'Name', 'conv1')
    reluLayer('Name','relu1')
    convolution2dLayer(2, 64, 'Padding','same', 'Stride', 1, 'Name', 'conv2')
    reluLayer('Name','relu2')
    convolution2dLayer(3, 64, 'Padding','same', 'Stride', 2, 'Name', 'conv4')
    reluLayer('Name','relu4')
    convolution2dLayer(3, 128, 'Padding','same', 'Stride', 2, 'Name', 'conv5')
    reluLayer('Name','relu5')
    convolution2dLayer(2, 256, 'Padding','same', 'Stride', 2, 'Name', 'conv6')
    reluLayer('Name','relu6')
    ];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    transposedConv2dLayer(1, 64, 'Cropping', 'same', 'Stride', 1, 'Name', 'traspos1')
    reluLayer('Name','ree1')
    transposedConv2dLayer(3, 256, 'Cropping', 'same', 'Stride', 2, 'Name', 'transpose1')
    reluLayer('Name','reelu1')
    transposedConv2dLayer(3, 128, 'Cropping', 'same', 'Stride', 2, 'Name', 'transpose2')
    reluLayer('Name','reflu2')
    transposedConv2dLayer(3, 64, 'Cropping', 'same', 'Stride', 2, 'Name', 'transpose3')
    reluLayer('Name','reldu3')
    transposedConv2dLayer(3, 64, 'Cropping', 'same', 'Stride', 2, 'Name', 'transpose4')
    reluLayer('Name','redlu4')
    transposedConv2dLayer(3, 64, 'Cropping', 'same', 'Stride', 2, 'Name', 'transpose5')
    reluLayer('Name','reslu5')
    transposedConv2dLayer(1, 3, 'Cropping', 'same', 'Name', 'transpose6')
    regressionLayer("Name","regressionoutput")
    ];

lgraph = addLayers(lgraph,tempLayers);
clear tempLayers;
lgraph = connectLayers(lgraph,"relu6","traspos1");



miniBatchSize  = 512;

input =fileDatastore(fullfile('input'),'ReadFcn',@load,'FileExtensions','.mat');
output=fileDatastore(fullfile('video'),'ReadFcn',@load,'FileExtensions','.mat');


inputDatat = transform(input,@(data) rearrange_datastore_input(data));
outputDatat = transform(output,@(data) rearrange_datastore_output(data));


trainData=combine(inputDatat,outputDatat);


% inp =fileDatastore(fullfile('inp_val'),'ReadFcn',@load,'FileExtensions','.mat');
% out=fileDatastore(fullfile('out_val'),'ReadFcn',@load,'FileExtensions','.mat');
%
%
% inputt = transform(inp,@(data) rearrange_datastore_input(data));
% outputt = transform(out,@(data) rearrange_datastore_output(data));
%
%
% valData=combine(inputt,outputt);

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',10000, ...
    'InitialLearnRate',0.5*1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',50, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','gpu',...
    'Verbose',true);




% here I defined my network architecture
% here I defined my training options
[net,info]=trainNetwork(trainData, lgraph, options);

%net=trainNetwork(trainData, layerGraph(net), options);
% for i=1:6
%     asd(i,:)=lagmatrix(asd(i,:),100);
% end
%asd(7:9,:)=repmat(asd(7:9,1),1,4999);
%asd(9,:)=linspace(0,4,4999);
%asd(1:6,:)=asd(1:6,1);

for i=1:10
    load('C:\Users\44772\Desktop\deep_vision_2\results_passive\input\1.mat')
    for j=7:9
        asd(j,:)=lagmatrix(asd(j,:),(i-1)*5);
    end
    %asd(i,:)=0;
    image = asd(:,1:end);
    image=reshape(image,[3,3,4999]);
    image=repmat(image,11);
    image=image(1:32,1:32,:);
    image=reshape(image,[32,32,1,4999]);
    ypred2 = predict(net,image);
    err(i,2)=immse(ypred2,YTrain);
end

image = asd(7:9,1:end);
image=reshape(image,[3,1,4999]);
image=repmat(image,32);
image=image(1:32,1:32,:);
image=reshape(image,[32,32,1,4999]);
ypred = predict(net,image);

%ypred = predict(net,trainData);
% asd=read (input);
%ypred = predict(net,asd.XTrain(:,:,:,1:900));
%deepNetworkDesigner(layers);

%imresize(ypred,[512 512]);

% act1 = activations(net,image(:,:,:,1:900),'transpose5');
%reshape(act1,128,128,1,900*64);


%
%act1 = activations(net,asd.XTrain(:,:,:,1:900),'Decoder-Stage-2-Conv-3');

%act1 = activations(net,XTrain(:,:,:,1:900),'Decoder-Stage-2-Conv-3');
% reshape(act1,128,128,1,900*64);
%asd=ans(:,:,1,69:64:end);

function image = rearrange_datastore_input(data)
image = data.asd(:,1:end);
image=reshape(image,[3,3,4999]);
image=repmat(image,11);
image=image(1:32,1:32,:);
image=reshape(image,[32,32,1,4999]);

image = data.asd(7:9,1:end);
image=reshape(image,[3,1,4999]);
image=repmat(image,32);
image=image(1:32,1:32,:);
image=reshape(image,[32,32,1,4999]);

image = num2cell(image, 1:3); % Wrap 1x21x1x100 data in 1x1x1x100 cell
image = image(:); % Reshape 1x1x1x100 cell to 1x100 cell
end

function image = rearrange_datastore_output(data)
image=data.YTrain;
image = num2cell(image, 1:3); % Wrap 1x21x1x100 data in 1x1x1x100 cell
image = image(:);
end


A=[ypred ypred2 YTrain];
v = VideoWriter('res_train.avi');

open(v)

%A=[];
%A=cat(4,A,[ypred;YTrain]);
writeVideo(v,mat2gray(A(:,:,:,:)));

%writeVideo(v,mat2gray(A(:,:,:,1:3000)));

close(v)


v = VideoWriter('vid.avi');
open(v)


writeVideo(v,mat2gray(YTrain(:,:,:,:)));



close(v)