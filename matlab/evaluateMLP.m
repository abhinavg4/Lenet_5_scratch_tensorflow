function [output output1] = evaluateMLP(activationFunction,Weights, Sample,bias,Weights_conv,Weights_bias,p,batch)
% Evaluates the MLP
%INPUT :- activationFunction, Weights :- weight matrix ; Sample :- input  ,
%Weights_conv  : Wieghts of conv layers , bias_conv : biases of conv layers
%p : just for plotting preferances, batch : for plotting
%example on which to evauate
%OUTPUT :- output final result vector after passing through network

    %Forward Pass
    noOfHiddenUnits = length(Weights)+1;
    ActualInput = cell(1,noOfHiddenUnits);
    ActualOutput = cell(1,noOfHiddenUnits);
    %inputVector = inputValues(:, n(k));
    input_img = transpose(vec2mat(Sample,28));
        y_1 = vl_nnconv(single(input_img), Weights_conv{1}, Weights_bias{1}, 'pad', 2) ;
        y_1_a = activationFunction(y_1);
        y_maxpool_1 = vl_nnpool(y_1_a,2,'stride',2);
        y_2 = vl_nnconv(y_maxpool_1,Weights_conv{2},Weights_bias{2});
        y_2_a = activationFunction(y_2);
        y_maxpool_2 = vl_nnpool(y_2_a,2,'stride',2);
        if(p)
            figure;
            str = sprintf('final conv vizualization 5x5 for digit %f',p-1);
            imagesc(y_maxpool_2(:,:,1));colormap gray;title(str);
            str1 = sprintf('f_conv_visual_b:%f_%f.png',batch,p-1);
            saveas(gcf,str1);
            close(gcf);
        end
            
        ActualInput{1} = [reshape(y_maxpool_2,[400,1,1]);1];
        
    
    %ActualInput{1} = Sample;
    %ActualInput{1} = inputValues(:, n);
    ActualOutput{1} = ActualInput{1};
    for j = 2 : noOfHiddenUnits
        ActualInput{j} = Weights{j-1}*ActualOutput{j-1};
        ActualOutput{j} = activationFunction(ActualInput{j});
        if(bias)
            if(j~=noOfHiddenUnits)
                ActualOutput{j}(end) = 1;
            end
        end
    end
   
    output = ActualOutput{noOfHiddenUnits};
    
    output1 = ActualOutput{noOfHiddenUnits - 1};
end