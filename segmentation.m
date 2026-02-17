%% Load flower data

imageDir = fullfile('daffodilSeg/ImagesRsz256');
labelDir = fullfile('daffodilSeg/LabelsRsz256');

imds = imageDatastore(imageDir);

classNames = ["Flower" "Background"];
labelIDs = {1, [2;3;4]};

pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);

[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionData(imds,pxds);

tbl = countEachLabel(pxds);
totalNumberOfPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalNumberOfPixels;
classWeights = 1./frequency


%% Supporting function in https://uk.mathworks.com/help/vision/ug/semantic-segmentation-using-deep-learning.html

function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionData(imds,pxds)
% Partition data by randomly selecting 80% of the data for training. The
% rest is used for testing and validation.
    
% Set initial random state for example reproducibility.
rng(0); 
numFiles = numpartitions(imds);
shuffledIndices = randperm(numFiles);

% Use 80% of the images for training.
numTrain = round(0.80 * numFiles);
trainingIdx = shuffledIndices(1:numTrain);

% Use 10% of the images for validation
numVal = round(0.10 * numFiles);
valIdx = shuffledIndices(numTrain+1:numTrain+numVal);

% Use the rest for testing.
testIdx = shuffledIndices(numTrain+numVal+1:end);

% Create image datastores for training and test.
imdsTrain = subset(imds,trainingIdx);
imdsVal = subset(imds,valIdx);
imdsTest = subset(imds,testIdx);

% Create pixel label datastores for training and test.
pxdsTrain = subset(pxds,trainingIdx);
pxdsVal = subset(pxds,valIdx);
pxdsTest = subset(pxds,testIdx);
end

%% Train CNN Model

inputSize = [256, 256, 3];

lgraph = layerGraph;

firstLayers = [
    imageInputLayer(inputSize)
    convolution2dLayer(7,32, "Stride",2,"Padding","same")
    maxPooling2dLayer(3, "Padding","same","Name","firstMaxPool","Stride",1)
    %convolution2dLayer(3,32, "Stride",1,"Padding","same")
    %maxPooling2dLayer(3, "Padding","same","Stride",1)
    %batchNormalizationLayer
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
    convolution2dLayer(3,64,"Name","conv3_5x5_","Padding","same")
    convolution2dLayer(3,64,"Padding","same")
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

% Upsampling and classification layer

upsamplingLayer = [
     transposedConv2dLayer(4,128,'Stride',2,"Name","upsample","Cropping","same")
     reluLayer
     transposedConv2dLayer(4,64,'Stride',2,'Cropping','same')
     reluLayer
     transposedConv2dLayer(4,32,'Stride',2,'Cropping','same')
     reluLayer
     convolution2dLayer(1,2,"Padding","same")
     softmaxLayer
     pixelClassificationLayer('Classes',tbl.Name,'ClassWeights',classWeights)];

lgraph = addLayers(lgraph, upsamplingLayer);
lgraph = connectLayers(lgraph, "block3_concat", "upsample");

val = combine(imdsVal, pxdsVal);

options = trainingOptions("sgdm",...
    LearnRateSchedule="piecewise",...
    LearnRateDropPeriod=6,...
    LearnRateDropFactor=0.005,...
    InitialLearnRate=1e-2,...
    Momentum=0.9,...
    L2Regularization=0.005,...
    ValidationData=val,...
    MaxEpochs=20,...  
    MiniBatchSize=8,...
    Shuffle="every-epoch",...
    CheckpointPath=tempdir,...
    VerboseFrequency=10,...
    ValidationPatience=4);

train = combine(imdsTrain, pxdsTrain);

net = trainNetwork(train, lgraph, options);
save('segmentnet.mat','net');


%% Test Network

load('segmentnet.mat');
reset(imdsTest);

testImage = read(imdsTest);
imshow(testImage)

segmentationTest = semanticseg(testImage, net);
segmentationTestOverlay = labeloverlay(testImage, segmentationTest);
imshow(segmentationTestOverlay)

pxdsResults = semanticseg(imdsTest,net,Classes=classNames,MiniBatchSize=8);
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,Verbose=false);
metrics.DataSetMetrics
metrics.ClassMetrics
metrics.NormalizedConfusionMatrix

classLabels = categorical(classNames);
cm = confusionchart(table2array(metrics.ConfusionMatrix), classLabels, 'Title','Confusion Matrix');

