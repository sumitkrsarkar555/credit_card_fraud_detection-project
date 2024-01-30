# Credit Card Fraud Detection using Machine Learning Project

## Table of Contents

- [Problem Statement](#problem-statement)
- [Business Understanding](#business-understanding)
- [Business Goal](#business-goal)
- [Dataset](#dataset)
    - [Dataset Information](#dataset-information)
- [Project Pipeline](#project-pipeline)
    - [Importing Libraries](#importing-libraries)
    - [Data Reading](#data-reading)
    - [Data Information](#data-information)
    - [Data Exploration and Visualization](#data-exploration-and-visualization)
    - [Data Preprocessing and Data Cleaning](#data-preprocessing-and-data-cleaning)
        - [Checking for Missing Values](#checking-for-missing-values)
        - [Standardizing the Amount Column](#standardizing-the-amount-column)
        - [Dropping the Time Column](#dropping-the-time-column)
        - [Checking for Duplicates](#checking-for-duplicates)
        - [Dropping Duplicate Values](#dropping-duplicate-values)
        - [Checking 0 and 1 Values of Class Column](#checking-0-and-1-values-of-class-column)
        - [Plotting Barplot of 0 and 1 Values](#plotting-barplot-of-0-and-1-values)
        - [Dropping NaN Values](#dropping-nan-values)
        - [Creating Variables for Train and Test Split](#creating-variables-for-train-and-test-split)
    - [Undersampling](#undersampling)
    - [Data Modeling and Model Training](#data-modeling-and-model-training)
        - [Creating 2 Dataframes - Normal and Fraud](#creating-2-dataframes---normal-and-fraud)
        - [Checking Dataframe Size](#checking-dataframe-size)
        - [Creating a New Dataframe](#creating-a-new-dataframe)
        - [Training the Model using Logistic Regression, Random Forest, Decision Tree](#training-the-model-using-logistic-regression-random-forest-decision-tree)
        - [Comparing the Results](#comparing-the-results)
        - [Creating ROC Curve](#creating-roc-curve)
        - [Creating Learning Curve Diagram](#creating-learning-curve-diagram)
    - [Oversampling](#oversampling)
        - [Creating Variables for Oversampling](#creating-variables-for-oversampling)
        - [Training the Model using Logistic Regression, Random Forest, Decision Tree](#training-the-model-using-logistic-regression-random-forest-decision-tree-1)
        - [Plotting Barplot and ROC Curve](#plotting-barplot-and-roc-curve)
        - [Saving the Model](#saving-the-model)
        - [User Input and Prediction](#user-input-and-prediction)
        - [Creating Learning Curve Diagram](#creating-learning-curve-diagram-1)

## Problem Statement

Credit card fraud is a major concern for both financial institutions and cardholders. Traditional rule-based methods are not always effective in detecting fraud, especially for new and sophisticated types of fraud. Machine learning algorithms can be trained on large datasets to learn patterns and detect anomalies in transactions, making them effective for detecting credit card fraud. By analyzing different features such as transaction amount, location, time, and user behavior, machine learning models can accurately identify fraudulent transactions and reduce the number of false positives. This helps financial institutions to prevent fraud and protect their customers from financial losses.

## Business Goal

Aims to prevent financial losses and provide high security to overall system.

## Dataset

This dataset is downloaded from Kaggle website.
- Link: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Dataset Information

The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise. Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.

## Project Pipeline

### Importing Libraries

Importing different libraries and dependencies to do this project on Google Colab.

### Data Reading

Uploading dataset downloaded from Kaggle website and copying the path of the dataset.

### Data Information

Showing the dataset head and tail function to see which columns are important and which are not. Describe function is used to describe the mean, min, and max function. Displaying columns and datatypes of each column.

### Data Exploration and Visualization

Exploring the dataset using visualization techniques.

### Data Preprocessing and Data Cleaning

- Check for missing values and count them for each column
- Standardize the amount column importing standard scaler function from sklearn
- Dropping the time column from dataframe
- Checking for duplicates if any 
- Dropping the duplicate values
- Checking the 0 and 1 value of class column
- Plotting a barplot of 0 and 1 value in graph
- Dropping NaN values from the dataset
- Creating variable x and y for test and train split 

### Undersampling

- Data Modeling and Model Training
- Creating 2 Dataframes - Normal and Fraud
- Checking Dataframe Size
- Creating a New Dataframe
- Training the Model using logistic regression, random forest classifier, decision tree classifier
- Comparing the Results (Accuracy, Precision, Recall, and F1 Score) using barplot
- Creating ROC Curve to compare each model
- Creating Learning Curve Diagram of each model

### Oversampling

- Creating x_res, y_res  2 variable using SMOTE (Synthetic Minority Over-sampling Technique) function 
- Training the Model using logistic regression, random forest, decision tree classifier 
- Plotting bar plot and ROC curve to choose best model 
- Importing joblib function and save the model as a credit_card_fraud_detection_model 
- Asking user to input any value 
- Printing if the value is normal or fraud transaction 
- Creating Learning Curve Diagram of each model 

### Saving the Model and Making Predictions

Now we save the model using joblib function. Ask the user for input, then the model automatically predicts and prints if the values are fraud or genuine transactions.

