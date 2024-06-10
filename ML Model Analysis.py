#Name: Sanjana Jagarlapudi

import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler #z-score normalization
from sklearn.preprocessing import MinMaxScaler #MinMax normalization
from sklearn.decomposition import PCA
import numpy as np

# Loading the Iris data set
iris = datasets.load_iris()
X = iris.data
y = iris.target

#Preproccessing the data (We are using z-score normalization here)
standard_scaler = StandardScaler()
X_StandardNormalized = standard_scaler.fit_transform(X) # Fit the normalization object to the data in order to transform it

#Some models need data that is pre-processed in a different way, thus we are using Min Max Normalization here
minMax_scaler = MinMaxScaler()
X_MinMaxNormalized = minMax_scaler.fit_transform(X) # Fit the normalization object to the data in order to transform it

#PCA to mitigate overfitting for XGBoost 
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_StandardNormalized) 

#Use 5-fold cross validation:
kfold = KFold(n_splits=5, shuffle=True, random_state=1)

#Making the evaluation metrics that will be used to score our models 
scoring = {
    'accuracy': make_scorer(accuracy_score), #use the make_scorer method listed in the api to easily make each evaluation metric 
    'f1_weighted': make_scorer(f1_score, average='weighted'), #this means that we weight the score of each class when using this metric
    'roc_auc_ovr': make_scorer(roc_auc_score, needs_proba=True, average='macro', multi_class='ovr') #needs_proba=True because this metric requires model predictions in a probability format, 
    #average='macro' tells the scorer to calculate scores independently for each class and then take the average while treating all classes equally, this is the functionality that we want 
    #for this program. Finally, multi_class='ovr' means "One-vs-Rest" and tells the scorer how the AUC should be calculated: by training a single classifier per class
}

# Making the models and storing them in a dictionary
models = {
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(probability=True), # Setting probability to true allows for methods like predict_proba to be used
    "RandomForest": RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=5, min_samples_split=15, min_samples_leaf=3, random_state=42), #hyper parameter tuning 
    "XGBoost": xgb.XGBClassifier(max_depth=3, n_estimators=100,gamma=1, eta=0.01, min_child_weight=3, reg_lambda=1, reg_alpha=0.5, subsample=0.5, colsample_bytree=0.7),
    "KNN": KNeighborsClassifier()
}

results = {} #initalize a dictionary to store our results

for name, model in models.items():
    #Different models need data that is preprocessed differently in order to optimize their performance and mitigate overfitting:
    if name == "XGBoost": #for example, the XGBoost model needs data that has been pre-processed with pca for the best results, while others don't
        crossValidation = cross_validate(model, X_pca, y, cv=kfold, scoring=scoring, return_train_score=True) 
    elif name == "KNN" : 
        crossValidation = cross_validate(model, X_MinMaxNormalized, y, cv=kfold, scoring=scoring, return_train_score=True) #using 5-fold cross validation to train and test each model and saving the scores
    elif (name == "SVM" or name == "RandomForest"): 
        crossValidation = cross_validate(model, X_StandardNormalized, y, cv=kfold, scoring=scoring, return_train_score=True)
    else: #otherwise, no preprocessing techniques needed
        crossValidation = cross_validate(model,X, y, cv=kfold, scoring=scoring, return_train_score=True)

    results[name] = {
        'Train Accuracy': np.mean(crossValidation['train_accuracy']), #find the mean of the scores of each model across all folds and 
        'Test Accuracy': np.mean(crossValidation['test_accuracy']),   #input the appropriate value into the results dictionary
        'Train F1 Score': np.mean(crossValidation['train_f1_weighted']),
        'Test F1 Score': np.mean(crossValidation['test_f1_weighted']),
        'Train ROC AUC': np.mean(crossValidation['train_roc_auc_ovr']),
        'Test ROC AUC': np.mean(crossValidation['test_roc_auc_ovr'])
    }

#Printing out the results
for model, scores in results.items():
    print(f"Model: {model}")
    print(f"  Train Acc: {scores['Train Accuracy']:.4f}")
    print(f"  Test Acc: {scores['Test Accuracy']:.4f}")
    print(f"  Train F1: {scores['Train F1 Score']:.4f}")
    print(f"  Test F1: {scores['Test F1 Score']:.4f}")
    print(f"  Train AUC: {scores['Train ROC AUC']:.4f}")
    print(f"  Test AUC: {scores['Test ROC AUC']:.4f}")
    print() 