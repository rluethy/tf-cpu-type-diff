# tf-cpu-type-diff
Demonstrate different results with Tensorflow on Intel and AMD CPUs

This program demonstrates that running identical machine learning processing on Intel and AMD CPUs yields slightly different results. 

The processing consists of training a binary MLP classifier using a training set and applying it to a validation set.
The input data is tabular with 7 numeric features. The code trains a simple MLP model and computes predicted probabilities when the model is applied to the validation set. The resulting differences are shown below:

![intel](https://user-images.githubusercontent.com/94056492/141120853-d479336d-d36f-443a-b629-64f6cb825c5c.png)
