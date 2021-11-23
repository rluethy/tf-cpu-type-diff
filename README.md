# Demonstrate different results with Tensorflow on Intel and AMD CPUs


This program demonstrates that running identical machine learning processing on Intel and AMD CPUs yields slightly different results.

The processing consists of training a binary MLP classifier using a training set and applying it to a validation set.

The input data is tabular with 29 numeric features.

The code trains a simple MLP model and computes predicted probabilities when the model is applied to the validation set.
The resulting differences are shown below:

![intel](https://user-images.githubusercontent.com/94056492/141120853-d479336d-d36f-443a-b629-64f6cb825c5c.png)


### Instructions to reproduce the different output files
1 start an EC2 instance with Intel CPU

  - we used a t3.medium instance type in region us-west-1 and AMI ami-053ac55bdcfe96e85 (ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-20211021)
  - log into instance and run the commands below to install the required libraries and run the test script:

    ```bash
    sudo apt update
    sudo apt -y install python3-pip
    sudo pip3 install tensorflow==2.5.0 pandas jupyter matplotlib
    git clone https://github.com/rluethy/tf-cpu-type-diff.git
    cd tf-cpu-type-diff/
    python3 TF_CPU_types_V3.py
    ```
  - download the output file `val_preds_Intel.csv`

2 start an EC2 instance with AMD CPU

  - for example t3a.medium instance type and AMI ami-053ac55bdcfe96e85
  - log into instance and run the same commands as above
  - download the output file `val_preds_AMD.csv`

3 compare the prob_1 column in files `val_preds_Intel.csv` from step 1 and `val_preds_AMD.csv` from step 2
