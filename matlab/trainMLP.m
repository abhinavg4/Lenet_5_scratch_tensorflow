function [Weights_conv Weights Weights_bias] = trainMLP(activationFunction, dActivationFunction,methodToUse, HiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate, momentum, b1, b2, epsi,bias,validata)
% trainMLP Creates a multi-layer perceptron
% and trains it on the MNIST dataset.
%
% INPUT:
% activationFunction             : Activation function used in both layers.
% dActivationFunction            : Derivative of the activation
%
% numberOfHiddenUnits            : Number of hidden units.
% inputValues                    : Input values for training (784 x 60000)
% targetValues                   : Target values for training (1 x 60000)
% epochs                         : Number of epochs to train.
% batchSize                      : Plot error after batchSize images.
% learningRate                   : Learning rate to apply.
%bias                            : Weather to use bias
%
% OUTPUT:
% Weights                        : cell of weights where Weights{i} is matrix of weights from layer i to i+1
% Wegihts_conv                   : Weights of conv network
% Weights_bias                   : biases of the conv network   

% The number of training vectors.
trainingSetSize = size(inputValues, 2);

% Input vector to fc layers has 401 ( 1bias) dimensions.
inputDimensions = 401;

% We have to distinguish 10 digits.
outputDimensions = size(targetValues, 1);

%Adding Bias
if(bias)
HiddenUnits(2:end-1)=HiddenUnits(2:end-1) + 1;
end


%adding Input and output layers to hidden layers
HiddenUnits = [inputDimensions HiddenUnits outputDimensions];
noOfHiddenUnits=length(HiddenUnits);

%Weights{i} denotes weight matrix from layer i to i+1
Weights = cell(1,noOfHiddenUnits-1);
Weights_conv = cell(1,2);
Weights_bias = cell(1,2);

rng(0,'twister');
Weights_conv{1} = sqrt(2/175).*randn(5,5,1,6,'single') ;
Weights_bias{1} = sqrt(1/784).*randn(6,1,'single');
rng(0,'twister');
Weights_conv{2} = sqrt(2/550).*randn(5,5,6,16,'single') ;
Weights_bias{2} = sqrt(1/1000).*randn(16,1,'single');

%used for Momentum
oldWeight = cell(1,noOfHiddenUnits-1);
oldWeight_conv = cell(1,2);
oldWeight_bias = cell(1,2);

% Initialize the weights and old weights.
for i = 1:noOfHiddenUnits-1
    rng(0,'twister');
    Weights{i} =   sqrt(2/(HiddenUnits(1,i+1)+HiddenUnits(1,i))).*randn(HiddenUnits(1,i+1), HiddenUnits(1,i));
    oldWeight{i} = zeros(HiddenUnits(1,i+1), HiddenUnits(1,i));
end

for i = 1:2
    oldWeight_conv{i} = 0.*Weights_conv{i};
    oldWeight_bias{i} = 0.*Weights_bias{i};
end

%DeltaWeight will store derivative of loss function w.r.t each weight
DeltaWeight = oldWeight;
DeltaWeight_conv = Weights_conv;
DeltaWeight_conv_add = Weights_conv;
DeltaWeight_bias = Weights_bias;
DeltaWeight_bias_add = Weights_bias;

% %To be used for Adam ; m : first momentum ; v = second_momentum ;
% m = oldWeight;
% v = oldWeight;
% %To be used for Adagrad
cache = oldWeight;
cache_c = oldWeight_conv;
cache_b = oldWeight_bias;

%%
% code for plotting 
figure; hold on;
str = sprintf('Train Error and Validation Error\n BatchSize = %f \n learning rate = %f \n',batchSize,learningRate);
title(str);
%xlabel('No of Examples seen/(5000)');
xlabel('No of steps');
ylabel('Error');
h = animatedline;
h.Color = 'r';
h.LineStyle = '--';
h.Marker = 'o';
z = animatedline;
z.Color = 'b';
z.LineStyle = ':';
z.Marker = 'x';
countq = 0;
error =0 ;
%%

for e = 1: epochs
    %learning rate decay
     if e==4
         learningRate = 0.1*learningRate
     end
     
     r = randperm(60000);
     counterr =0 ;
     T = idivide(60000,int16(batchSize));
     
for t = 1: T
    
    batch = r(1,double(t-1)*batchSize+1:double(t)*batchSize);
    %reset all delta Weights to 0
    for layer = 1 : length(DeltaWeight)
        DeltaWeight{layer} = 0*DeltaWeight{layer};
    end
    
    %code for weight decay
    %for layer = 1 : length(Weights)
     %   Weights{layer} = (0.9).*Weights{layer};
    %end
    %for layer = 1: 2
     %   Weights_conv{layer} = (0.9).*Weights_conv{layer};
    %end
    
    for convn = 1:2
        DeltaWeight_conv{convn} = 0*DeltaWeight_conv{convn};
        DeltaWeight_bias{convn} = 0*DeltaWeight_bias{convn};
        DeltaWeight_conv_add{convn} = 0*DeltaWeight_conv_add{convn};
        DeltaWeight_bias_add{convn} = 0*DeltaWeight_bias_add{convn};
    end
    
    %ActualInput{i} is Wx+b to the layer
    %ActualOutput{i} is sigma(Wx+b)
    ActualInput = cell(1,noOfHiddenUnits);
    ActualOutput = cell(1,noOfHiddenUnits);

    %Store BackProp Error
    BackPropDelta = cell(1,noOfHiddenUnits-1);
    
    %error=0;
    
    for k = batch

        % Propagate the input vector through the network.
        % Forward Pass
        
        [input_img y_1 y_1_a y_1_maxpool y_2 y_2_a y_2_maxpool]= Forward(inputValues(:,k),Weights_conv,activationFunction,Weights_bias);
        
        ActualInput{1} = [reshape(y_2_maxpool,[400,1,1]) ;1];
        ActualOutput{1} = ActualInput{1};
   
        for j = 2 : noOfHiddenUnits
            ActualInput{j} = Weights{j-1}*ActualOutput{j-1};
            ActualOutput{j} = activationFunction(ActualInput{j});
            %We don't modify bias nodes
            if(bias)
                if(j~=noOfHiddenUnits)
                    ActualOutput{j}(end) = 1;
                end
            end
        end
        %Applying Softmax
        expo = sum(exp(ActualOutput{noOfHiddenUnits}));
        for q = 1 : length(ActualOutput{noOfHiddenUnits})
            ActualOutput{noOfHiddenUnits}(q) = exp( ActualOutput{noOfHiddenUnits}(q))./expo;
        end

        targetVector = targetValues(:, k);
        error=error+(0.5)*norm(ActualOutput{noOfHiddenUnits} - targetVector);
        countq = countq+1;
        
        %Backward Pass
        BackPropDelta{noOfHiddenUnits-1}=dActivationFunction(ActualInput{noOfHiddenUnits}).*(ActualOutput{noOfHiddenUnits} - targetVector);
        for j= noOfHiddenUnits-2:-1:1
            BackPropDelta{j} = dActivationFunction(ActualInput{j+1}).*(Weights{j+1}'*BackPropDelta{j+1});
        end
        
        %backpropogation for conv layers
        temp = Weights{1}'*BackPropDelta{1};
        dzdxf = reshape(temp(1:400),[5,5,16]);
        dzdy_maxpool_2 = repelem(dzdxf,2,2).*(repelem(y_2_maxpool,2,2)==y_2_a);
        %dz_ml_2r = vl_nnpool(y_2_a,2,dzdxf,'stride',2);
        dy_2 = drev_Activation(y_2).*dzdy_maxpool_2;
        %dy_2r = vl_nnsigmoid(y_2,dzdy_maxpool_2);
        [dx_s2 DeltaWeight_conv_add{2} DeltaWeight_bias_add{2}] = conv(y_1_maxpool,Weights_conv{2},[],dy_2);
        %[dx_s2r DeltaWeight_conv_addr2 dbr] = vl_nnconv(y_maxpool_1,Weights_conv{2},[],dy_2);
        dx_c1 = repelem(dx_s2,2,2).*(repelem(y_1_maxpool,2,2)==y_1_a);
        %dx_c1r = vl_nnpool(y_1_a,2,dx_s2,'stride',2);
        dy_c1 = drev_Activation(y_1).*dx_c1; 
        %dy_c1r = vl_nnsigmoid(y_1,dx_c1);
        [qq DeltaWeight_conv_add{1} DeltaWeight_bias_add{1}] = conv(padarray(single(input_img),[2 2]), Weights_conv{1}, [],dy_c1) ;
        %[qqr DeltaWeight_conv_addr1 dbr] = vl_nnconv(single(input_img), Weights_conv{1}, [],dy_c1, 'pad', 2) ;
      
        
        %finally calculate delta as O(i)*BackPropDelta(j)
        for j = 1 : noOfHiddenUnits-1
            DeltaWeight{j} = DeltaWeight{j}+BackPropDelta{j}*ActualOutput{j}';
        end
        for j = 1:2
            DeltaWeight_conv{j} = DeltaWeight_conv{j} + DeltaWeight_conv_add{j};
            DeltaWeight_bias{j} = DeltaWeight_bias{j} + DeltaWeight_bias_add{j};
        end
        
    end

    if(methodToUse==2)
        for j = 1: noOfHiddenUnits-1
            DeltaWeight{j} = DeltaWeight{j}./batchSize;
            m{j} = b1.*m{j}+(1-b1).*DeltaWeight{j};
            v{j} = b2.*v{j}+(1-b2).*(DeltaWeight{j}.^2);
            mh = m{j}./(1-b1^t);
            vh = v{j}./(1-b2^t);
            Weights{j} = Weights{j} - (learningRate.*mh)./(vh.^(0.5)+epsi);
        end
    elseif methodToUse==3
        for j = 1:noOfHiddenUnits-1
            DeltaWeight{j} = DeltaWeight{j}./batchSize;
            cache{j} = cache{j} + DeltaWeight{j}.^2;
            Weights{j} = Weights{j} - learningRate.*DeltaWeight{j}./(cache{j}.^(0.5) + epsi);
        end
      
        for j = 1 :2
            DeltaWeight_conv{j} = DeltaWeight_conv{j}./batchSize ;
            cache_c{j} = cache_c{j} + DeltaWeight_conv{j}.^2;
            Weights_conv{j} = Weights_conv{j} - learningRate.*DeltaWeight_conv{j}./(cache_c{j}.^(0.5) + epsi);
            DeltaWeight_bias{j} = DeltaWeight_bias{j}./batchSize ;
            cache_b{j} = cache_b{j} + DeltaWeight_bias{j}.^2;
            Weights_bias{j} = Weights_bias{j} - learningRate.*DeltaWeight_bias{j}./(cache_b{j}.^(0.5) + epsi);
        end
    else
        for j = 1 : noOfHiddenUnits-1
            DeltaWeight{j} = DeltaWeight{j}./batchSize;
            oldWeight{j} = momentum.*oldWeight{j} + learningRate.*DeltaWeight{j};
            Weights{j} = Weights{j} - oldWeight{j};
        end
        
        for j = 1:2
            DeltaWeight_conv{j} = DeltaWeight_conv{j}./batchSize;
            oldWeight_conv{j} = momentum.*oldWeight_conv{j} + learningRate.*DeltaWeight_conv{j};
            Weights_conv{j} = Weights_conv{j} - oldWeight_conv{j};
            DeltaWeight_bias{j} = DeltaWeight_bias{j}./batchSize;
            oldWeight_bias{j} = momentum.*oldWeight_bias{j} + learningRate.*DeltaWeight_bias{j};
            Weights_bias{j} = Weights_bias{j} - oldWeight_bias{j};
        end
        
    end
    %error = error/batchSize;
    %addpoints(h,counterr+(e-1)*12,double(error));
    %plot(t+(e-1)*double(T),error,'-');
    %drawnow;
    %error for plotting
    
   %plotting for each 5000 batch
   if((double(t)*batchSize)>((5000)*double(counterr)))
    counterr = counterr+1;
    errortp = error/(countq);
    addpoints(h,counterr+(e-1)*12,double(errortp));
    %plot(t+(e-1)*double(T),error,'*');
    drawnow;
    error = 0;countq=0;

    
    batch = validata;
    errori=0;
    for k = batch
        
        [input_img y_1 y_1_a y_1_maxpool y_2 y_2_a y_2_maxpool]= Forward(inputValues(:,k),Weights_conv,activationFunction,Weights_bias);
        ActualInput{1} = [reshape(y_2_maxpool,[400,1,1]) ;1];
        ActualOutput{1} = ActualInput{1};


        for j = 2 : noOfHiddenUnits
            ActualInput{j} = Weights{j-1}*ActualOutput{j-1};
            ActualOutput{j} = activationFunction(ActualInput{j});
            %We don't modify bias nodes
            if(bias)
                if(j~=noOfHiddenUnits)
                    ActualOutput{j}(end) = 1;
                end
            end
        end

        %Applying Softmax
        expo = sum(exp(ActualOutput{noOfHiddenUnits}));
        for q = 1 : length(ActualOutput{noOfHiddenUnits})
            ActualOutput{noOfHiddenUnits}(q) = exp( ActualOutput{noOfHiddenUnits}(q))./expo;
        end

        targetVector = targetValues(:, k);

        errori=errori+(0.5)*norm(ActualOutput{noOfHiddenUnits} - targetVector);
    end
    errori = errori/2000;
    addpoints(z,counterr+(e-1)*12,double(errori));
    %plot(counterr+e-1,error,'c*');
    drawnow;
    
    
        end;
         
    end;
end;

end
