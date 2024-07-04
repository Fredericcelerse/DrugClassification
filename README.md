# DrugClassification
This introductory project demonstrates how to build and use ML approaches for drug classification.

In this example, we show how to construct an AI model based on the Support Vector Machine (SVM) architecture and enhance its accuracy and efficiency using Bayesian Optimization and probability approaches.

## Prerequisites

### Anaconda

To execute the code, we will set up a specific environment using Anaconda. To install it, visit [Anaconda Installation](https://docs.anaconda.com/free/anaconda/install/).

### Setup conda environment

First, create the conda environment:
```
conda create -n DrugClassification python=3.8
```

Then, activate the conda environment:
```
conda activate DrugClassification
```

Once the environment is properly created, install the necessary Python libraries to execute the code:
```
pip install scikit-learn scikit-optimize matplotlib pandas seaborn joblib
```

### Database

The database used in this project, named "Kaggle: Drugs A, B, C, X, Y for Decision Trees", is coming from the following website: https://www.kaggle.com/datasets/pablomgomez21/drugs-a-b-c-x-y-for-decision-trees/data 
Once you have downloaded the database, ensure that you have a directory that contains the file [drug200.csv](drug200.csv).

## Project architecture

This example consists of two main parts:

***1. Building a SVM model from sratch***   
***2. Deploying the model for applications***

Let us see in more details these two aspects

### 1. Building a SVM model from sratch

There is a script named [train.py](train.py). After verifying the path to the database, you can launch the script by typing:
```
python train.py
```
The model will first load the entire database, then pre-process the data (especially the strings), and then set up and train the SVM model.    

The SVM parameters are automatically optimized using the Bayesian Optimization approach, where we define manually the research space.   

Importantly, the model is trained in order to provide uncertainty in each prediction. We also plot the confusion matrix to have an idea about the reliability of the training (see [Confusion_Matrix.png](Confusion_Matrix.png)) and performed 5 cross-validations. 

More details about the setup and how the model is built are available within the script itself.   

### 2. Deploying the model for applications

At the end of the script, the model is saved in a file named [svm_model.joblib](svm_model.joblib). This model can be then used by launching the script [predict_drug.py](predict_drug.py):
```
python predict_drug.py
```
By entering manually the data you have, you can obtain a prediction with uncertainty. For instance, imagine we have are a woman of 25 years old, with a low blood pressure and cholesterol, and with a ratio of Sodium/Potassium of 20, we will have:
```
Enter the desired features:
Age: 25
Sex (M/F); F
Blood Pressure (HIGH/LOW/NORMAL): LOW
Cholesterol (HIGH/LOW): LOW
Na_to_K: 20
Prediction: drugY
Probability: [2.36455441e-03 6.91262030e-04 1.09479980e-02 1.46437103e-02 9.71352475e-01]
```

In this case, the model predict for us to take the drug Y, with a probability of 97%. Thus feature can be used in the future to find automatically where the predictions failed to be accurate, offering an elegant and automatic way to reinforce the database for future investigations. 
