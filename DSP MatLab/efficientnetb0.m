pathToImages = "C:\Users\Nooq\Desktop\DSP projekt\Lego Dataset";

Lego_ds = imageDatastore(pathToImages,"IncludeSubfolders",true,LabelSource="foldernames");


[trainImgs,testImgs] = splitEachLabel(Lego_ds,0.8,"randomized");

AtrainImgs = augmentedImageDatastore([224 224],trainImgs, 'ColorPreprocessing','gray2rgb');
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


options = trainingOptions("sgdm","InitialLearnRate",0.01, "MiniBatchSize",32,"MaxEpochs",1,"Momentum",0.9,"Verbose",1,"Plots","training-progress");



[Lego_EffNet,info] = trainNetwork(AtrainImgs,lgraph,options);
save Lego_EffNet;

%%

testpreds = classify(Lego_EffNet,AtestImgs);

nnz(testpreds == testImgs.Labels)/numel(testpreds);

confusionchart(testImgs.Labels,testpreds);

%%
test_image = "C:\Users\Nooq\Desktop\750x750.png";
figure
imshow(test_image)
%%
load Lego_EffNet;

%%

%for photo_nr=1:1
numImages = numel(Lego_ds.Labels);
idx = randperm(numImages,1);
I = imtile(Lego_ds,Frames=idx);
%figure
%imshow(I)

image = I;



    lego1 = imresize(image, [224 224]);
    %figure
    %imshow(lego1)
    %title(classify(Lego_EffNet,lego1))
    %classify(Lego_EffNet,lego1)
    figure
    subplot(1,2,1);
imshow(I);
subplot(1,2,2);
imshow(lego1);

    %%imshow(lego1)
   title(classify(Lego_EffNet,lego1))
%end

