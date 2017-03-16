# cats_vs_dogs
This is the implementation to solve Kaggle's cat vs dog segregation.

- This is the VGG16 Implementation of [Kaggle Cat_vs_Dogs](https://www.kaggle.com/c/dogs-vs-cats) competition.
- Network is initialized by pretrained ImageNet [weights](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz). Note: The weights are transformed from Caffe and for convenience are taken from [here](https://www.cs.toronto.edu/~frossard/post/vgg16/)
- The total training dataset (25000) is divided into two sets 
1. Train (22500 = 90% of original train) 
2. Validation (2499 = 10% of original train)

- Two trainings are done with small changes in augmentations. The Network achieves near perfect accuracy in training in both cases, however validation accuracy differs a bit.
- Two CSV submission files are provided on Test set (12500 images) for evaluation.
- For detailed analysis of implementation, go through the Notebooks.

# Submissions
1. [1st Submission](https://www.cs.toronto.edu/~frossard/post/vgg16/)
2. [2nd Submission](https://www.cs.toronto.edu/~frossard/post/vgg16/)


# Training progress for First run
![alt text](https://github.com/gondal1/cats_vs_dogs/blob/master/images/run1.png)

# Training progress for Second run
![alt text](https://github.com/gondal1/cats_vs_dogs/blob/master/images/run1.png)
