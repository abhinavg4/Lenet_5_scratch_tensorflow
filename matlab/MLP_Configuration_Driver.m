 function [] = MLP_Configuration()

    % Load MNIST.
    inputValues = loadMNISTImages('train-images.idx3-ubyte');
    labels = loadMNISTLabels('train-labels.idx1-ubyte');
    bias = 1;

    % Transform the labels to correct target values.
    targetValues = 0.*ones(10, size(labels, 1));
    for n = 1: size(labels, 1)
        targetValues(labels(n) + 1, n) = 1;
    end
%%
    % Input Hidden Layes Here. Eg:- For 2 layers with 10, 20 nodes input = [10 20]
    HiddenUnits = [120 84];
%%
    % Choose appropriate parameters. 0.001 for adam and 0.01 for adagrad
    learningRate = 10^-3;
    
    %if GD with mometum is used
    momentum = 0.9;
    %if ADM is use
    b1 = 0.9; b2 = 0.999;epsi=10^-8;
    %methodToUse ; {1 : GD with momentum} ; {2 : Adam} ; {3 : Adagrad} ;
    %use only momentum for now
    methodToUse = 3;
%%

    % Choose activation function. Which Activation to use can be set in
    % Acitvation.m and derv_Activation.m
    activationFunction = @Activation;
    dActivationFunction = @drev_Activation;

    % Load validation set.
    inputValuesi = loadMNISTImages('t10k-images.idx3-ubyte');
    labelsi = loadMNISTLabels('t10k-labels.idx1-ubyte');
%%
    % Choose batch size for batch update and epochs = number of iterations.
    batchSize = [16 32 64 128];
    epochs = 3;
    r = randperm(60000);
    fprintf('Train lenet5 with %d hidden layers.\n', length(HiddenUnits));
    fprintf('Learning rate: %d.\n', learningRate);
    for b = batchSize
      learningRate = sqrt(b/8)*learningRate
    [Weights_conv Weights Weights_bias] = trainMLP(activationFunction, dActivationFunction, methodToUse, HiddenUnits, inputValues, targetValues, epochs, b, learningRate, momentum,b1,b2,epsi,bias,r(1,1:2000));

    
    fprintf('Testing:\n');

    [correctlyClassified, classificationErrors] = testMLP(activationFunction, Weights, inputValuesi, labelsi,bias,Weights_conv,Weights_bias,b);


    fprintf('Classification errors: %d\n', classificationErrors);
    fprintf('Correctly classified: %d\n', correctlyClassified);
    end
end
