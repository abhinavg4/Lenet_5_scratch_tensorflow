function [correctlyClassified, classificationErrors] = validateMLP(activationFunction, Weights, inputValues, labels,bias,Weights_conv,Weights_bias,batch)
% Validate the MLP using the
% validation set.
%
% INPUT:
% activationFunction             : Activation function to be used
% Weights,Weights_conv           : Weights of the Layers and conv_network
% inputValues                    : Input values for training (784 x 10000).
% labels                         : Labels for validation (1 x 10000).
% bias,Weights_bias              : Weather to use bias and conv_biases
%
% OUTPUT:
% correctlyClassified            : Number of correctly classified values.
% classificationErrors           : Number of classification errors.
%
    tsne_m = [];
    yes = 0.*ones(1,10);
    testSetSize = size(inputValues, 2);
    classificationErrors = 0;
    correctlyClassified = 0;
    %iterate over all test to see which correctly classifies
    for n = 1: testSetSize
        inputVector = inputValues(:, n);
        if(yes(int16(labels(n)+1)))
            p = 0;
        else
            p=labels(n)+1 ;
            yes(int16(labels(n)+1)) = labels(n)+1;
        end
        %evaluate and compare
        [outputVector tsne_a]= evaluateMLP(activationFunction, Weights, inputVector, bias,Weights_conv,Weights_bias,p,batch);

        [m class] = max(outputVector);
        %class = decisionRule(outputVector);
        if class == labels(n) + 1
            correctlyClassified = correctlyClassified + 1;
        else
            classificationErrors = classificationErrors + 1;
        end;
        tsne_m = [tsne_m;tsne_a'];
%         if(p)
%             figure;
%            str = sprintf('final fc vizualization 12x7 for digit %f',labels(n));
%             imagesc(reshape(tsne_m(n,:),[12,7]));colormap gray;title(str);
%             str1 = sprintf('f_fc_visual_b:%f_%f.png',batch,labels(n));
%             saveas(gcf,str1);
%             close(gcf);
%         end
    end
%     mapx = tsne(tsne_m(1:5000,:),[],2,50,30);
%     figure ;
%     gscatter(mapx(:,1),mapx(:,2),labels(1:5000));
end
