pathToImages = "C:\Users\Nooq\Desktop\DSP projekt\Lego Dataset";

Lego_ds = imageDatastore(pathToImages,"IncludeSubfolders",true,LabelSource="foldernames");


[trainImgs,testImgs] = splitEachLabel(Lego_ds,0.8,"randomized");

AtrainImgs = augmentedImageDatastore([224 224],testImgs, 'ColorPreprocessing','gray2rgb');
AtestImgs = augmentedImageDatastore([224 224], testImgs, 'ColorPreprocessing','gray2rgb');
Alego_ds = augmentedImageDatastore([224 224], Lego_ds);


numClasses = numel(categories(Lego_ds.Labels));
categories(Lego_ds.Labels)


net=efficientnetb0;
lgraph = layerGraph(net);
net.Layers


newFc = fullyConnectedLayer(numClasses,"Name","New_fc");
lgraph = replaceLayer(lgraph,"efficientnet-b0|model|head|dense|MatMul",newFc);
newOut = classificationLayer("Name","new_out");
lgraph = replaceLayer(lgraph,"classification",newOut);


options = trainingOptions("sgdm","InitialLearnRate",0.01, "MiniBatchSize",32,"MaxEpochs",50,"Momentum",0.9,"Verbose",1,"Plots","training-progress");

[Birds_EffNet,info] = trainNetwork(AtrainImgs,lgraph,options);
save Lego_EffNet;

