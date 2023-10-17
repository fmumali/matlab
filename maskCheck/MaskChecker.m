classdef MaskChecker
    properties
        ImageFolder
        Imds
        ImdsTrain
        ImdsValidation
        Net
        NewNet
        InputSize
    end
    
    methods
        function obj = MaskChecker(imageFolder)
            obj.ImageFolder = imageFolder;
        end
        
        function obj = loadData(obj)
            obj.Imds = imageDatastore(obj.ImageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders', true);
            [obj.ImdsTrain, obj.ImdsValidation] = splitEachLabel(obj.Imds, 0.7);
        end
        
        function sampleImage = displaySampleImage(obj)
            sampleImage = readimage(obj.ImdsTrain, randi(numel(obj.ImdsTrain.Files)));
            imshow(sampleImage);
        end
        
        function obj = prepareNetwork(obj)
            obj.Net = googlenet;
            obj.InputSize = obj.Net.Layers(1).InputSize;
        end
        
        function obj = transferLearning(obj)
            lgraph = layerGraph(obj.Net);
            numClasses = numel(categories(obj.ImdsTrain.Labels));
            
            newLearnableLayer = fullyConnectedLayer(numClasses, ... 
                'Name', 'new_fc', ...
                'WeightLearnRateFactor', 10, ...
                'BiasLearnRateFactor', 10);

            lgraph = replaceLayer(lgraph, 'loss3-classifier', newLearnableLayer); 
            newClassLayer = classificationLayer('Name', 'new_classoutput'); 
            lgraph = replaceLayer(lgraph, 'output', newClassLayer);

            options = trainingOptions('sgdm', ...
                'MiniBatchSize', 10, ...
                'MaxEpochs', 6, ...
                'InitialLearnRate', 3e-4, ...
                'Plots', 'training-progress');

            augimdsTrain = augmentedImageDatastore(obj.InputSize(1:2), obj.ImdsTrain);
            obj.NewNet = trainNetwork(augimdsTrain, lgraph, options);
        end
        
        function accuracy = validate(obj)
            augimdsValidation = augmentedImageDatastore(obj.InputSize(1:2), obj.ImdsValidation);
            [YPred, probs] = classify(obj.NewNet, augimdsValidation);
            accuracy = mean(YPred == obj.ImdsValidation.Labels);
        end
        
        function displayValidationImages(obj)
            [YPred, probs] = classify(obj.NewNet, augmentedImageDatastore(obj.InputSize(1:2), obj.ImdsValidation));
            idx = randperm(numel(obj.ImdsValidation.Files), 4); 
            figure;
            for i = 1:4
                subplot(2, 2, i);
                I = readimage(obj.ImdsValidation, idx(i));
                imshow(I);
                label = YPred(idx(i));
                title(string(label) + ", " + num2str(100 * max(probs(idx(i), :)), 3) + "%");
            end
        end
    end
end
