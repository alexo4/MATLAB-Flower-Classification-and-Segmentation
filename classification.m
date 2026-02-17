
%% Organises the images of each flower into their own folders for labelling if this is not yet done

fileDir = "17flowers/";
organiseFiles(fileDir);

function organiseFiles(fileDir)

if isfolder("17flowers/1")
    return;
end

files = dir(fullfile(fileDir));
class = 1;

for idx = 2:80:size(files)
   mkdir(fileDir + string(class));
   for j = 1:80

       if idx + j > size(files)
           break
       end
       movefile(fileDir + string(files(idx + j).name), fileDir + string(class));
       
    end

    class = class + 1;

end

end

%% Load images as resized and split them into a train test split

imds = imageDatastore(fileDir, "IncludeSubfolders",true,"LabelSource",'foldernames');
imds.ReadFcn = @customRead;

% image resize code taken from internet - https://www.mathworks.com/matlabcentral/answers/449046-how-can-i-resize-image-stored-in-imagedatastore
function data = customRead(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
data = imresize(data,[256,256]);
end

train_split = 0.8;
valTest_split = 0.1;

[imdsTrain,imdsVal,imdsTest] = splitEachLabel(imds,train_split,valTest_split);


%% Train CNN Model
numClasses = 17;
inputSize = [256, 256, 3];

lgraph = layerGraph;

firstLayers = [
    imageInputLayer(inputSize)
    convolution2dLayer(7,32, "Stride",2)
    maxPooling2dLayer(3, "Padding","same","Name","firstMaxPool","Stride",2)
    convolution2dLayer(3,32, "Stride",1)
    maxPooling2dLayer(3, "Padding","same","Stride",2)
    batchNormalizationLayer
    convolution2dLayer(1,32,"Name","stem")];

% Block 1

block1_branch_1x1 = [convolution2dLayer(1,64,"Name","block1_conv1_1x1")
    reluLayer("Name","block1_relu1x1")];

block1_branch_3x3 = [
    convolution2dLayer(1,64,"Name","block1_conv2_1x1")
    convolution2dLayer(3,64,"Name","conv2_3x3_","Padding","same")
    reluLayer("Name","block1_conv2_3x3")];

block1_branch_5x5 = [
    convolution2dLayer(1,64,"Name","block1_conv3_1x1")
    convolution2dLayer(5,64,"Name","conv3_5x5_","Padding","same")
    reluLayer("Name","block1_conv3_5x5")];

block1_branch_pool = [
    maxPooling2dLayer(3,"Name","block1_maxPool","Padding","same")
    convolution2dLayer(1,64,"Name","conv4_1x1_")
    reluLayer("Name","block1_conv4_1x1")];

block1_concatLayer = depthConcatenationLayer(4, 'Name', 'block1_concat');
block1_maxPooling = maxPooling2dLayer(3, "Padding","same","Name","secondMaxPool","Stride",2);

lgraph = addLayers(lgraph, firstLayers);
lgraph = addLayers(lgraph, block1_branch_1x1);
lgraph = addLayers(lgraph, block1_branch_3x3);
lgraph = addLayers(lgraph, block1_branch_5x5);
lgraph = addLayers(lgraph, block1_branch_pool);
lgraph = addLayers(lgraph, block1_concatLayer);
lgraph = addLayers(lgraph, block1_maxPooling);

lgraph = connectLayers(lgraph, "stem","block1_conv1_1x1");
lgraph = connectLayers(lgraph, "stem","block1_conv2_1x1");
lgraph = connectLayers(lgraph, "stem","block1_conv3_1x1");
lgraph = connectLayers(lgraph, "stem","block1_maxPool");

lgraph = connectLayers(lgraph, "block1_relu1x1","block1_concat/in1");
lgraph = connectLayers(lgraph, "block1_conv2_3x3","block1_concat/in2");
lgraph = connectLayers(lgraph, "block1_conv3_5x5","block1_concat/in3");
lgraph = connectLayers(lgraph, "block1_conv4_1x1","block1_concat/in4");

lgraph = connectLayers(lgraph, "block1_concat","secondMaxPool");

% Block 2

block2_branch_1x1 = [convolution2dLayer(1,128,"Name","block2_conv1_1x1")
    reluLayer("Name","block2_1x1Relu")];

block2_branch_3x3 = [
    convolution2dLayer(1,128,"Name","block2_conv2_1x1")
    convolution2dLayer(3,128,"Name","conv2_3x3_2_","Padding","same")
    reluLayer("Name","block2_conv2_3x3")];

block2_branch_5x5 = [
    convolution2dLayer(1,128,"Name","block2_conv3_1x1")
    convolution2dLayer(3,128,"Name","conv3_5x5_2_","Padding","same")
    convolution2dLayer(3,128,"Padding","same")
    reluLayer("Name","block2_conv3_5x5")];

block2_branch_pool = [
    maxPooling2dLayer(3,"Name","block2_maxPool","Padding","same")
    convolution2dLayer(1,128,"Name","conv4_1x1_2_")
    reluLayer("Name","block2_conv4_1x1")];

block2_concatLayer = depthConcatenationLayer(4, 'Name', 'block2_concat');
block2_maxPooling = maxPooling2dLayer(3, "Padding","same","Name","thirdMaxPool","Stride",2);

lgraph = addLayers(lgraph, block2_branch_1x1);
lgraph = addLayers(lgraph, block2_branch_3x3);
lgraph = addLayers(lgraph, block2_branch_5x5);
lgraph = addLayers(lgraph, block2_branch_pool);
lgraph = addLayers(lgraph, block2_concatLayer);
lgraph = addLayers(lgraph, block2_maxPooling);

lgraph = connectLayers(lgraph, "secondMaxPool","block2_conv1_1x1");
lgraph = connectLayers(lgraph, "secondMaxPool","block2_conv2_1x1");
lgraph = connectLayers(lgraph, "secondMaxPool","block2_conv3_1x1");
lgraph = connectLayers(lgraph, "secondMaxPool","block2_maxPool");

lgraph = connectLayers(lgraph, "block2_1x1Relu","block2_concat/in1");
lgraph = connectLayers(lgraph, "block2_conv2_3x3","block2_concat/in2");
lgraph = connectLayers(lgraph, "block2_conv3_5x5","block2_concat/in3");
lgraph = connectLayers(lgraph, "block2_conv4_1x1","block2_concat/in4");

lgraph = connectLayers(lgraph, "block2_concat","thirdMaxPool");

% Block 3

block3_branch_1x1 = [convolution2dLayer(1,256,"Name","block3_conv1_1x1")
    reluLayer("Name","block3_1x1Relu")];

block3_branch_3x3 = [
    convolution2dLayer(1,256,"Name","block3_conv2_1x1")
    convolution2dLayer(3,256,"Padding","same")
    reluLayer("Name","block3_conv2_3x3")];

block3_branch_5x5 = [
    convolution2dLayer(1,256,"Name","block3_conv3_1x1")
    convolution2dLayer(5,256,"Padding","same")
    reluLayer("Name","block3_conv3_5x5")];

block3_branch_pool = [
    maxPooling2dLayer(3,"Name","block3_maxPool","Padding","same")
    convolution2dLayer(1,256)
    reluLayer("Name","block3_conv4_1x1")];

block3_concatLayer = depthConcatenationLayer(4, 'Name', 'block3_concat');

lgraph = addLayers(lgraph, block3_branch_1x1);
lgraph = addLayers(lgraph, block3_branch_3x3);
lgraph = addLayers(lgraph, block3_branch_5x5);
lgraph = addLayers(lgraph, block3_branch_pool);
lgraph = addLayers(lgraph, block3_concatLayer);

lgraph = connectLayers(lgraph, "thirdMaxPool","block3_conv1_1x1");
lgraph = connectLayers(lgraph, "thirdMaxPool","block3_conv2_1x1");
lgraph = connectLayers(lgraph, "thirdMaxPool","block3_conv3_1x1");
lgraph = connectLayers(lgraph, "thirdMaxPool","block3_maxPool");

lgraph = connectLayers(lgraph, "block3_1x1Relu","block3_concat/in1");
lgraph = connectLayers(lgraph, "block3_conv2_3x3","block3_concat/in2");
lgraph = connectLayers(lgraph, "block3_conv3_5x5","block3_concat/in3");
lgraph = connectLayers(lgraph, "block3_conv4_1x1","block3_concat/in4");

% output and classifying layers

outputLayers = [
    batchNormalizationLayer("Name","outputLayer")
    globalAveragePooling2dLayer
    dropoutLayer(0.5)
    flattenLayer
    fullyConnectedLayer(17)
    softmaxLayer
    classificationLayer];

lgraph = addLayers(lgraph, outputLayers);
lgraph = connectLayers(lgraph, "block3_concat", "outputLayer");

options = trainingOptions("sgdm", 'MaxEpochs', 30, 'ValidationData',imdsVal, ...
    'ValidationFrequency',25,'Verbose',false,'Plots','training-progress', ...
    'miniBatchSize',128,'L2Regularization',0.0001);

net = trainNetwork(imdsTrain, lgraph, options);

save("classnet.mat","net")

%% Metrics  

load("classnet.mat")

YPred = classify(net, imdsTest);

Ytest = imdsTest.Labels;

accuracy = mean(YPred == Ytest)

cm = confusionchart(Ytest, YPred);

