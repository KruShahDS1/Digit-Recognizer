# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 18:07:42 2021

@author: kshah
"""

####### DIGIT RECOGNIZER DATA SET - FORM DIGITS ################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 

          # library for creating visualizations
#################################################################
from sklearn.feature_selection import f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


df = pd.read_csv('trainfeatures42k.csv')

class_label = 'Label'
features = df.columns.to_list()
features.remove(class_label)

######### Data Processing ##################################################
def mahalanobis_dist(X: np.ndarray, mean: np.ndarray, cov: np.ndarray):
   """ Calculate the Mahalanobs distance for an array of n samples and m dimensions."""
   n = X.shape[0]
   X_c = X - mean
   X_cov_inv = np.linalg.inv(cov)
   left = np.matmul(X_c, X_cov_inv)
   # Below we calculate X_c*X_cov_inv*X_c'
   # size n x n
   dist_squared_matrix = np.matmul(left, X_c.T)
   diag = np.diagonal(dist_squared_matrix)
   return diag
 ##############################################################################  
   
    

################## Finding Outliers using Mahanobis distance ######################
def find_outliers(X: np.ndarray, p:int) -> np.ndarray:
    X_mean = np.mean(X, axis=0)
    X_cov  = np.cov(X, rowvar = False)
    dist_squared = mahalanobis_dist(X,X_mean,X_cov)
    idx = np.flip(np.argsort(dist_squared)).tolist()
    dist_sorted = np.take(dist_squared, idx).tolist()
    return dist_sorted[0:p], idx[0:p]

def normalize(X: np.ndarray, a:float, b: float):
    min = X.min(axis = 0).T
    max = X.max(axis = 0).T
    X_norm = ((X - min)/(max - min))* (b-a)+a
    return X_norm

def fishers_LDR (labels,X:np.ndarray):
    classes = np.unique(labels)
    k = len(classes)
    m = X.shape[1]
    
    #Calculate standard deviation per class per feature
    
    std = np.zeros([k,m])
    for cl in classes:
        std[cl] = np.std(X[np.where(labels[:]==cl)[0],:], axis = 0)
    total_std = np.sum(std, axis =0)
    mean = np.zeros([k,m])
    W_temp = np.zeros([m,0])
    
    for cl in classes:
        mean[cl] = np.mean(X[np.where(labels[:] == cl)[0],:], axis = 0)
        mean_div = np.divide(mean[cl],total_std).reshape(-1,1)
        W_temp = np.concatenate((W_temp , mean_div), axis = 1)
    
    ind_temp = np.empty([m,0],dtype = int)
    cor_temp = np.empty([m,0])
    
    for j in range(0,k):
        WW_aux = np.concatenate((W_temp[:,0:j], W_temp[:, j+1:k]),axis = 1)
        WW_cur = np.repeat(W_temp[:,j].reshape(-1,1),k-1, axis =1)
        WW = (np.subtract(WW_cur,WW_aux))
        rankW = np.min(WW, axis = -1).reshape(-1,1)
        rankW_abs = -1 *np.abs(rankW.T)
        ind_temp = np.concatenate((ind_temp,np.argsort(rankW_abs).T), axis = 1)
        cor_temp = np.concatenate((cor_temp,np.sort(rankW_abs).T),axis = 1)
        
    ind_temp = ind_temp.T.flatten('F').reshape(-1,1).T    
    cor_temp = cor_temp.T.flatten('F').reshape(-1,1).T  

    cor_sort = np.sort(cor_temp)
    index = np.argsort(cor_temp)
    index_sort = ind_temp.take(index).reshape(1,-1)
    
    ind_temp2 = np.fliplr(index_sort)
    cor_temp2 = np.fliplr(cor_sort)
    
    u, v = np.unique(ind_temp2, return_index = True)
    w = np.sort(v).reshape(-1,1)
    s = np.argsort(v).reshape(-1,1)
    values = np.fliplr((cor_temp2.take(w)).T)
    rank_index = np.fliplr((u.take(s)).T)
    return values.flatten(), rank_index.flatten()

#############################################################################

# find possible outliers per class (digit)
n_to_remove = 2 # number of outliers to remove (p in find_outliers)
outliers_dist = []
outliers_idx = []
for cl in  range(10):
    class_data = df.loc [df[class_label] == cl] 
    #Xclass =  class_data.to_numpy()[:,1:]
    class_dist, class_idx = find_outliers(class_data.to_numpy()[:,1:], n_to_remove) # drop label and index columns
    
    df_idx = class_data.iloc[class_idx].index.tolist()
    outliers_dist = outliers_dist + class_dist
    outliers_idx  = outliers_idx + df_idx
    
outliers_df = df.iloc[outliers_idx].copy().drop(features, axis = 1) 
outliers_df['mahalanobis']  = outliers_dist

# print the list of possible outliers and their mahalanobis distance
print(outliers_df)
# remove outliers
df.drop(outliers_idx, inplace = True) 

X = df.iloc[:, 1:61].to_numpy()
X = normalize(X, 0,1) 
labels = df[class_label].to_numpy(dtype = int).reshape(-1,1)
norm_data_with_labels = np.hstack((labels,X))
df_norm = pd.DataFrame(norm_data_with_labels, columns = df.columns.to_list())

vals, ranks_idx = fishers_LDR(labels,X)
ranked_features = np.array(features).take(ranks_idx)
print("The list of ranked features is : ", ranked_features)  

###################################################################################################
########### K-Fold Cross Validation #########################################################


def kfold_cross_validation(num_data: int, num_folds: int, shuffle: bool):
    
    if shuffle :
        data = np.random.permutation(num_data)
    else:
        data = np.arange(num_data)
        
    test_data_size = math.ceil(num_data/num_folds)
    total_size = num_folds*test_data_size

    part = np.zeros((total_size,))
    part = range(num_data) 
   # fill in the missing data with random samples
    if (total_size < num_data):
        part[total_size:num_data] = np.random.sample(range(1, num_data),total_size-num_data)
        
   #partition data for each fold . this will be the base for our test data  
    part = np.reshape(part,(num_folds,test_data_size))
    
    test = []
    training = []
    for i in range(num_folds):
        test.append(data[part[i]])
        #Get the indices that are not part of the test data partition which makes the training data
        indices = part[[j for j in range (num_folds) if j != i]].flatten()
        training.append(data[indices])
    return test, training    

# Run the  K-fold cross validation
num_folds = 4
# Get the test and training data
test,training = kfold_cross_validation(X.shape[0], num_folds, True)

  # Prepare Training and test sets
X_train = []
X_test = []
y_train = []
y_actual = []

top_features = ranked_features[0:30]  #Change to increase/reduce the number of features

for i in range(num_folds):
    X_train.append(df_norm.loc[training[i],top_features].to_numpy())
    X_test.append(df_norm.loc[test[i],top_features].to_numpy())
    y_train.append(df_norm.loc[training[i],[class_label]].to_numpy(dtype=int).ravel())
    y_actual.append(df_norm.loc[test[i],[class_label]].to_numpy(dtype=int).ravel())                                

print('length of X_train', len(X_train[:][0]))
print('length of X_test', len(X_test[:][0]))
print('length of y_train', len(y_train[:][0]))
print('length of y_actual', len(y_actual[:][0]))

###########################################################################################

#######  CLASSIFIERS RUN on the Processed Data

############################################################################################

class GBayesClassifier:
    
    def __init__(self):
        self.train_prob_membership = None
        self.train_variances = None
        self.train_means = None
        self.num_features = None
        self.num_classes = None
        
    def train(self, x , y):
        classes = np.unique(y)
        
        self.num_classes = len(classes)
        self.num_features = x.shape[1]
        self.train_prob_membership = np.zeros(self.num_classes)
        self.train_variances = np.zeros((self.num_classes,self.num_features))
        self.train_means = np.zeros((self.num_classes,self.num_features))
        
        for idx, cls in enumerate(classes):
            self.train_means[idx] = x[np.where(y == cls)].mean(axis=0)
            self.train_variances[idx] = x[np.where(y == cls)].var(axis=0)
            self.train_prob_membership[idx] = len(np.where(y == cls)[0]) / len(y)

    def predict_prob(self,x):
       
       num_obs, num_feats = x.shape

       probs = np.zeros((num_obs, self.num_classes)) 
       for obs in range(num_obs):
           for cls in range(self.num_classes):
               p = self.train_prob_membership[cls]
               t1 = 1 / np.sqrt(2 * np.pi * self.train_variances[cls])
               t2 = (x[obs] - self.train_means[cls]) ** 2 / self.train_variances[cls]
               t3 = np.exp(-0.5 * t2)
               t4 = (t1 * t3).prod()
               probs[obs, cls] = t4 *p
           probs[obs,:] = probs[obs, :] / np.sum(probs[obs,:])
                                                 
       return probs

    def predict (self,x):
        return np.argmax(self.predict_prob(x),axis=1)
     
###############################################################################################

figure, ax = plt.subplots(nrows = 1, ncols = num_folds, figsize = (25,10))
min_accuracy = 100
bayes_report = {}


for i in range(num_folds):
    
    gbc = GBayesClassifier()
    gbc.train(X_train[i], y_train[i])
    y_pred = gbc.predict(X_test[i])
    
    # gnb = GaussianNB()               # This is sklearn in-built Gaussian NAive Bayes Classifier 
    # gnb.fit(X_train[i], y_train[i])
    # y_pred = gnb.predict(X_test[i])
      
    cm = confusion_matrix(y_actual[i], y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix =cm)
    #f1_score(y_actual, y_pred, labels=np.unique(y_pred))
    
    precision = precision_score(y_actual[i], y_pred, average = 'micro',  zero_division=0)
    print('Precision score: {0:0.2f}'.format(precision))

    recall = recall_score(y_actual[i], y_pred,average = 'micro', zero_division=0)
    print('Recall score: {0:0.2f}'.format(recall))

    f1 = f1_score(y_actual[i], y_pred, average = 'micro', zero_division=0)
    print('f1 score: {0:0.2f}'.format(f1))
    
    accuracy = accuracy_score(y_actual[i], y_pred, normalize = False)
    print('Accuracy: {0:0.2f}'.format(accuracy/100))
    #accuracy = (len(y_pred[y_pred == y_actual[i]])/len(y_actual[i])) *100
    #Keep the Classification report for the worst fold 
    if (accuracy < min_accuracy):
        min_accuracy = accuracy
        bayes_report = classification_report(y_actual[i], y_pred, output_dict = True)
        print(bayes_report)
    ax[i].set_title("Custom, Fold" + str(i) + " Accuracy: "  +str(accuracy/100))
    disp.plot(ax = ax[i],colorbar = False)
    
    
###############################################################################    
#         Support Vector Machine                                              #
###############################################################################
figure, ax = plt.subplots(nrows = 1, ncols = num_folds, figsize = (25,10))
min_accuracy = 100
bayes_report = {}


for i in range(num_folds):
    
  
   SVM = SVC(kernel='linear', random_state = 234)
   SVM.fit(X_train[i],y_train[i])
   y_pred = SVM.predict(X_test[i])
   
  
   cm = confusion_matrix(y_actual[i], y_pred)
   disp = ConfusionMatrixDisplay(confusion_matrix =cm)
    #f1_score(y_actual, y_pred, labels=np.unique(y_pred))
    
   precision = precision_score(y_actual[i], y_pred, average = 'micro',  zero_division=0)
   print('Precision score: {0:0.2f}'.format(precision))

   recall = recall_score(y_actual[i], y_pred,average = 'micro', zero_division=0)
   print('Recall score: {0:0.2f}'.format(recall))

   f1 = f1_score(y_actual[i], y_pred, average = 'micro', zero_division=0)
   print('f1 score: {0:0.2f}'.format(f1))
    
   accuracy = accuracy_score(y_actual[i], y_pred, normalize = False)
   print('Accuracy: {0:0.2f}'.format(accuracy/100))
    #accuracy = (len(y_pred[y_pred == y_actual[i]])/len(y_actual[i])) *100
    #Keep the Classification report for the worst fold 
   if (accuracy < min_accuracy):
       min_accuracy = accuracy
       bayes_report = classification_report(y_actual[i], y_pred, output_dict = True)
       print(bayes_report)
   ax[i].set_title("Custom, Fold" + str(i) + " Accuracy: "  +str(accuracy/100))
   disp.plot(ax = ax[i],colorbar = False)
    




    