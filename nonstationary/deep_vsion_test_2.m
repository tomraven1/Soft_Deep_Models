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
    imageInputLayer([128 128 3],'Name','input_encoder','Normalization','none')
    convolution2dLayer(3, 64, 'Padding','same', 'Stride', 1, 'Name', 'conv1')
    reluLayer('Name','relu1')
    convolution2dLayer(2, 64, 'Padding','same', 'Stride', 1, 'Name', 'conv2')
    reluLayer('Name','relu2')
    convolution2dLayer(3, 64, 'Padding','same', 'Stride', 2, 'Name', 'conv4')
    reluLayer('Name','relu4')
    convolution2dLayer(3, 128, 'Padding','same', 'Stride', 2, 'Name', 'conv5')
    reluLayer('Name','relu5')
    convolution2dLayer(2, 128, 'Padding','same', 'Stride', 4, 'Name', 'ceonv6')
    reluLayer('Name','r3elu6')
    convolution2dLayer(2, 64, 'Padding','same', 'Stride', 2, 'Name', 'conv6')
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



miniBatchSize  = 128*1;

input =fileDatastore(fullfile('inp_man'),'ReadFcn',@load,'FileExtensions','.mat');
output=fileDatastore(fullfile('tar_man'),'ReadFcn',@load,'FileExtensions','.mat');


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
    'InitialLearnRate',1*1e-3, ...
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
for j=1:55
    image = XTrain;

    ypred = predict(net,image);
end

%ypred = predict(net,trainData);
% asd=read (input);
%ypred = predict(net,asd.XTrain(:,:,:,1:900));
%deepNetworkDesigner(layers);

%imresize(ypred,[512 512]);

 act1 = activations(net,asd.XTrain(:,:,:,1:900),'transpose5');
reshape(act1,128,128,1,900*64);
%
%act1 = activations(net,asd.XTrain(:,:,:,1:900),'Decoder-Stage-2-Conv-3');

%act1 = activations(net,XTrain(:,:,:,1:900),'Decoder-Stage-2-Conv-3');
% reshape(act1,128,128,1,900*64);
asd=ans(:,:,1,69:64:end);

function image = rearrange_datastore_input(data)

image = data.XTrain;
image = num2cell(image, 1:3); % Wrap 1x21x1x100 data in 1x1x1x100 cell
image = image(:); % Reshape 1x1x1x100 cell to 1x100 cell

% image = data.asd(:,1:end);
% image=reshape(image,[3,3,4999]);
% image=repmat(image,11);
% image=image(1:32,1:32,:);
% image=reshape(image,[32,32,1,4999]);
%
% % image = data.asd(7:9,1:end);
% % image=reshape(image,[3,1,4999]);
% % image=repmat(image,32);
% % image=image(1:32,1:32,:);
% % image=reshape(image,[32,32,1,4999]);
%
% image = num2cell(image, 1:3); % Wrap 1x21x1x100 data in 1x1x1x100 cell
% image = image(:); % Reshape 1x1x1x100 cell to 1x100 cell
end

function image = rearrange_datastore_output(data)
image=data.YTrain;
image = num2cell(image, 1:3); % Wrap 1x21x1x100 data in 1x1x1x100 cell
image = image(:);
end


A=[ypred ypred2 YTrain];
% v = VideoWriter('res1.avi');
%
% open(v)
%
% %A=[];
% %A=cat(4,A,[ypred;YTrain]);
% writeVideo(v,mat2gray(A(:,:,:,:)));
%
% %writeVideo(v,mat2gray(A(:,:,:,1:3000)));
%
% close(v)

% siz=50;
%
% for i = 1:990
%     [net,YPred(:,i)] = predictAndUpdateState(net,XTrain(:,:,:,i),'ExecutionEnvironment','gpu');
% end
%
% % layers_1(1,1).Mean=0;
%  net2 = assembleNetwork(layers_1);
% %
%  Ysc = reshape(YPred,[1,1,siz,990]);
%  ypred2 = predict(net2, Ysc );

k=1;

ha = tight_subplot(4,2,[.01 .03],[.1 .01],[.01 .01]);


for j=[2,4,14,37]

%2 4 14 37
     baseFileName = fullfile('inp_man',[num2str(j),'.mat']);
     load(baseFileName)
    baseFileName = fullfile('tar_man',[num2str(j),'.mat']);
    load(baseFileName)

    image = XTrain;

    ypred = predict(net,image);
   A=[ YTrain(:,:,:,1)];
   B=[ypred(:,:,:,end)  YTrain(:,:,:,end) ];
      %  err3(j)=immse(repmat(YTrain(:,:,:,1),[1 1 1 990]),YTrain);

      %subplot(4,3,k)
      axes(ha(k))
      imshow(A)
      %subplot(4,3,k+1:k+2)
      axes(ha(k+1))
      imshow(B)
      k=k+2;
end





act1 = activations(net,XTrain(:,:,:,1:900),'transpose5');
%reshape(act1,128,128,1,900*64);

%reshape(act1,128*8,128*8,1,900);
%=ans(:,:,1,69:64:end);

for i=1:8
    for j=1:8
    asd((i-1)*128+1:(i)*128,(j-1)*128+1:j*128,:)=act1(:,:,i*j:64:end,:);
    end
end



act1 = activations(net,XTrain(:,:,:,1:900),'transpose4');
%reshape(act1,128,128,1,900*64);

%reshape(act1,128*8,128*8,1,900);
%=ans(:,:,1,69:64:end);

for i=1:8
    for j=1:8
    asd2((i-1)*64+1:(i)*64,(j-1)*64+1:j*64,:)=act1(:,:,i*j:64:end,:);
    end
end

imshow(asd(:,:,end))
figure;
imshow(asd2(:,:,end))