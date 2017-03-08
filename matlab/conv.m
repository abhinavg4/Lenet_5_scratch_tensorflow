function [dx dw db] = conv(x,w,b,dy)
%%
%conv function to give backprogation delta of weights, biases, x

%INPUT

%x is an array of dimension H x W x C x N 
%where (H,W) are the height and width of the image stack,
%C is the number of feature channels, and N is the number of images in the batch.

%w is an array of dimension FW x FH x FC x K 
%where (FH,FW) are the filter height and width and 
%K the number o filters in the bank. 
%FC is the number of feature channels in each filter and must match the number of feature channels C in X.

%b are the biases, dy is the derivative of the output layer

%OUTPUT
%dx : derivative of loss w.r.t input layers
%dw : dervative of loss w.r.t weight matrix
%db : derivative of loss w.r.t biases

%note: dx will have same dimension as x. dw will have same dimensions as w.
%db will have same dimension as b
%%    
    dw = w;
    x = single(x);
    w = single(w);
    b = single(b);
    dy = single(dy);
    %layer wise computing dw by convolving input with dy
    for i = 1:size(w,4)
        for j = 1:size(w,3)
           dw(:,:,j,i)=vl_nnconv(x(:,:,j),dy(:,:,i),[]);
        end
    end
    %dx is obtained after convolving dy with rotated w
    dx = vl_nnconv(dy,rot90(permute(w,[1,2,4,3]),2),[],'pad',size(w,1)-1);
    %db is sum of all detas if dy
    db = permute(sum(sum(dy,1),2),[3 1 2]);
end
