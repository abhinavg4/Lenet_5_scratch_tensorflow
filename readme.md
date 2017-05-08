# LeNet-5 From Scratch and using tensorflow

This repository contains code for LeNET-5 from scratch in Matlab and also using tensorflow. This was build as an assignment in my course CS698U.

## Problem Statement for the assignment

* Implement and train the LeNet-5 CNN for MNIST digit classification task

    1. From scratch e.g., using only Numpy, Matlab.
    2. Using a deep learning library; one of Caffe, Torch, Tensorflow, Keras

* Compare your results of two implementations above with each other.

* In the report:
    1. Compare the time taken by the conv layers vs. the fc layers
    2. Compare the number of params in the conv layers vs. the fc layers
    3. Visualize the features extracted from a randomly selected test example for each digit class and show t-SNE plots.
    4. Plot the training and validation error rates vs. the number of iterations
    5. Explore the effect of different batch sizes (16, 32, 64, 128) on training

* An iteration is one mini-batch, an epoch is a pass over the whole training data

### Project Details

* matlab :- This folder contains implementation of Le Net 5 from strach in matlab. README.pdf contains much of the details for this implementation. Also to run this you must have matconvnet installed as for convolution this code uses convnn.m .All parameters and other driver code is present in this file. To run the code :-
```
MLP_configuration_driver.m
```

* tensorflow :- This folder contains code for LeNet -5 using tensorflow. Much of the code is taken from tensorflow without a Phd tutorials.

### Running

To run the MLP you need to run MLP_configuration_driver.m 


```
MLP_configuration_driver.m
```

## Testing and accuracy

All plots form various tests can be found in plot folder. I have achieved accuracy of around 99.2% with this implementation.

## License

This project is licensed under the MIT License

## Acknowledgments

* Gaurav Sharma (Course Instructor CS698U)
