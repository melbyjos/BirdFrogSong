"""
This module contains classes for exploring and comparing various classifier models for labeled bird
and frog song datasets input as numpy arrays. These datasets can be high-dimensional embeddings or
vectorized spectrograms of audio clips produced prior to using this module.
"""
import numpy as np
import os
import librosa
import keras
import pandas as pd
import time
import random
import pickle

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from matplotlib.pyplot import cm

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


class Classifier():
    """Class for applying various classification algorithms to a dataset (X, y) input as a pair of
    numpy arrays. Models include Logistic Regression, SVM, KNN, Random Forests, and combinations of
    these with PCA. 
    """
    def __init__(self, 
                 input_X, 
                 input_y, 
                 stratify_var = None, 
                 shuffle = True, 
                 seed = True,
                 val_size = 0.15,
                 n_splits = 5,
                 kfold_shuff = True,
                 ordered_labels = []): 
        """Initialize the Classifier class with input data and other optional parameters. 

        Args:
            input_X (ndarray): numpy array of data X with corresponding labels y
            input_y (ndarray): numpy array of labels y
            stratify_var (list, optional): List of features along which to stratify the train test
                                            split. Defaults to None.
            shuffle (bool, optional): Toggle whether to shuffle the train test split. Defaults to True.
            seed (bool, optional): Toggle to seed at the beginning for every algorithm called on
                                    data. Defaults to True.
            val_size (float, optional): validation size for the train test split. Defaults to
                                        0.15/1.0. 
            n_splits (int, optional): number of kfold splits for each cross validation. Defaults to 5.
            kfold_shuff (bool, optional): toggle whether to shuffle in the kfold split. 
                                            Defaults to True.
            ordered_labels (list, optional): list of labels from which values of y are pulled. When
                                             a nontrivial list of labels is given, a confusion
                                             matrix is construced for each
                                             non-cross validated algorithm call. Defaults to [].
        """
        
        self.X = input_X
        self.y = input_y
        self.stratify_var = stratify_var
        self.shuffle = shuffle
        self.test_size = val_size
        self.n_splits = n_splits

        if seed:
            self.rand_state = 997
        else:
            self.rand_state = random.randint(55, 995)

        self.kfold_splits = n_splits
        self.kfold_shuff = kfold_shuff
        self.kfold = KFold(n_splits=self.kfold_splits,
                           shuffle=self.kfold_shuff,
                           random_state=self.rand_state)
        self.labels = ordered_labels

        self.total_statistics = {}  ## This will be a log of the important attributes and statistical results of the models.

    def t_t_split(self):
        """Generates a train test split of the input data. 
        """
        self.ttsplit_tup = train_test_split(self.X,
                                            self.y,
                                            shuffle=self.shuffle,
                                            random_state=self.rand_state,
                                            test_size=self.test_size) 
        
        self.X_train = self.ttsplit_tup[0]
        self.X_test = self.ttsplit_tup[1]
        self.y_train = self.ttsplit_tup[2]
        self.y_test = self.ttsplit_tup[3]

        self.total_statistics['ttsplit'] = self.ttsplit_tup
        
    def cross_val(self):
        """Generates a kfold split of the training data."""
        self.kfold_split = self.kfold.split(self.X_train, self.y_train)

    def log_reg(self, 
                CV = True,
                penalty = None,
                C_val = 1.0):
        """perform logistic regression on the training and test data. This method prints progress
        updates, statistics, sets appropriate attributes for the class instance, and stores results
        and statistical data in the class instance statistics dictionary. 

        Args:
            CV (bool, optional): Toggle whether or not to do cross validation. Defaults to True.
            penalty (str, optional): Penalty for the logistic regression. Defaults to None.
            C_val (float, optional): coefficient for the penalty function. Defaults to 1.
        """

        if CV:
            st = time.time()

            ## Makes a 1-D array of to record the accuracies in
            self.log_reg_test_accs = np.zeros(self.n_splits)
            self.log_reg_train_accs = np.zeros(self.n_splits)

            i = 0
            for train_index, test_index in self.kfold.split(self.X_train, self.y_train):
                ## Keeps track of the cross validation split you are on:

                print("CV Split: %s" % i)
                
                ##split off portions of the training data

                tt = self.X_train[train_index]
                ho = self.X_train[test_index]
                y_tt = self.y_train[train_index]
                y_ho = self.y_train[test_index]
                
                ## Run through scaler
                scaler = StandardScaler()

                ## fit the training data and transform that and holdout
                scaler.fit(tt)
                tt_scaled = scaler.transform(tt)
                ho_scaled = scaler.transform(ho)

                ##log reg model
                log_reg = LogisticRegression(max_iter=10000,
                                             penalty=penalty, 
                                             C=C_val)
                    
                ## fit the model
                log_reg.fit(tt_scaled, y_tt)

                #save accuracy for both test and training sets
                    
                self.log_reg_test_accs[i] = accuracy_score(y_ho, log_reg.predict(ho_scaled))
                self.log_reg_train_accs[i] = accuracy_score(y_tt, log_reg.predict(tt_scaled))

                i = i + 1

            ##Print out some progress and statistics:
            et = time.time() - st
            print("Elapsed Time: %s" % et)     
            
            self.log_reg_test_acc_max = np.max(self.log_reg_test_accs)
            print("The highest CV Logistic Regression test accuracy was", self.log_reg_test_acc_max)

            self.log_reg_train_acc_max = np.max(self.log_reg_train_accs)
            print("The highest CV Logistic Regression train accuracy was", self.log_reg_train_acc_max)

            ## Log everything into the stats dict

            self.total_statistics['log_reg_CV'] = {'test_acc': self.log_reg_test_accs, 
                                                   'train_acc': self.log_reg_train_accs, 
                                                   'test_max': self.log_reg_test_acc_max,
                                                   'train_max': self.log_reg_train_acc_max}

        else:

            ## Perform a single iteration of log reg

            ## Run through scaler
            scaler = StandardScaler()
            scaler.fit(self.X_train)
            self.X_train_scaled = scaler.transform(self.X_train)
            reg = LogisticRegression(max_iter=10000,
                                     penalty=penalty, 
                                     C=C_val).fit(self.X_train_scaled, self.y_train)

            self.X_test_scaled = scaler.transform(self.X_test)

            self.y_log_reg_test_pred = reg.predict(self.X_test_scaled)
            self.y_log_reg_train_pred = reg.predict(self.X_train_scaled)

            self.log_reg_test_acc = accuracy_score(self.y_test, self.y_log_reg_test_pred)
            print("The Logistic Regression test accuracy was", self.log_reg_test_acc)
            self.log_reg_train_acc = accuracy_score(self.y_train, self.y_log_reg_train_pred)
            print("The Logistic Regression train accuracy was", self.log_reg_train_acc)

            self.total_statistics['log_reg'] = {'test_acc': self.log_reg_test_acc,
                                                'train_acc': self.log_reg_train_acc}
            
            self.log_reg_test_pred_arr = np.concatenate((self.y_test.reshape(-1,1), self.y_log_reg_test_pred.reshape(-1,1)), axis = 1)

            if len(self.labels) > 0:
                self.log_reg_mat = np.zeros((len(self.labels) + 1, len(self.labels) + 1), dtype = int)
                self.log_reg_mat[0, 1:] = self.labels
                self.log_reg_mat[1:,0] = self.labels

                for item in self.log_reg_test_pred_arr:
                    test = item[0]
                    ind = np.where(self.log_reg_mat[0] == test)[0][0]
                    pred = item[1]
                    ind_pred = np.where(self.log_reg_mat[:,0] == pred)[0][0]
                    if test == pred:
                        self.log_reg_mat[ind, ind] += 2
                    else:
                        self.log_reg_mat[ind, ind] += 1
                        self.log_reg_mat[ind_pred, ind] += 1

                self.total_statistics['log_reg_mat'] = self.log_reg_mat
                print("You can load and save the confusion matrix from the pickled dictionary under the key 'log_reg_mat'")

    def svm(self, 
            CV = True):
        """Trains and tests a SVC on the training and test data. This method prints progress
        updates, statistics, sets appropriate attributes for the class instance, and stores results
        and statistical data in the class instance statistics dictionary. 

        Args:
            CV (bool, optional): Toggle whether or not to do cross validation. Defaults to True.
        """

        if CV:
            st = time.time()

            ## Makes a 1-D array of to record the accuracies in
            self.svm_test_accs = np.zeros(self.n_splits)
            self.svm_train_accs = np.zeros(self.n_splits)

            i = 0
            for train_index, test_index in self.kfold.split(self.X_train, self.y_train):
                ## Keeps track of the cross validation split you are on

                print("CV Split: %s" % i)
                tt = self.X_train[train_index]
                ho = self.X_train[test_index]
                y_tt = self.y_train[train_index]
                y_ho = self.y_train[test_index]
                
                scaler = StandardScaler()
                scaler.fit(tt)
                tt_scaled = scaler.transform(tt)
                ho_scaled = scaler.transform(ho)
                svm = SVC()
                    
                svm.fit(tt_scaled, y_tt)
                    
                self.svm_test_accs[i] = accuracy_score(y_ho, svm.predict(ho_scaled))
                self.svm_train_accs[i] = accuracy_score(y_tt, svm.predict(tt_scaled))

                i = i + 1
            et = time.time() - st
            print("Elapsed Time: %s" % et)     
            
            self.svm_test_acc_max = np.max(self.svm_test_accs)
            print("The highest CV SVM test accuracy was", self.svm_test_acc_max)

            self.svm_train_acc_max = np.max(self.svm_train_accs)
            print("The highest CV SVM train accuracy was", self.svm_train_acc_max)

            self.total_statistics['svm_CV'] = {'test_acc': self.svm_test_accs, 
                                               'train_acc': self.svm_train_accs, 
                                               'test_max': self.svm_test_acc_max,
                                               'train_max': self.svm_train_acc_max}

        else:

            ## Perform a single iteration of svm

            ## Run through scaler
            scaler = StandardScaler()
            scaler.fit(self.X_train)
            self.X_train_scaled = scaler.transform(self.X_train)
            svm = SVC().fit(self.X_train_scaled, self.y_train)

            self.X_test_scaled = scaler.transform(self.X_test)

            self.y_svm_test_pred = svm.predict(self.X_test_scaled)
            self.y_svm_train_pred = svm.predict(self.X_train_scaled)

            self.svm_test_acc = accuracy_score(self.y_test, self.y_svm_test_pred)
            print("The SVM test accuracy was", self.svm_test_acc)
            self.svm_train_acc = accuracy_score(self.y_train, self.y_svm_train_pred)
            print("The SVM train accuracy was", self.svm_train_acc)

            self.total_statistics['svm'] = {'test_acc': self.svm_test_acc,
                                            'train_acc': self.svm_train_acc}
            
            self.svm_test_pred_arr = np.concatenate((self.y_test.reshape(-1,1), self.y_svm_test_pred.reshape(-1,1)), axis = 1)

            if len(self.labels) > 0:
                self.svm_mat = np.zeros((len(self.labels) + 1, len(self.labels) + 1), dtype = int)
                self.svm_mat[0, 1:] = self.labels
                self.svm_mat[1:,0] = self.labels

                for item in self.svm_test_pred_arr:
                    test = item[0]
                    ind = np.where(self.svm_mat[0] == test)[0][0]
                    pred = item[1]
                    ind_pred = np.where(self.svm_mat[:,0] == pred)[0][0]
                    if test == pred:
                        self.svm_mat[ind, ind] += 2
                    else:
                        self.svm_mat[ind, ind] += 1
                        self.svm_mat[ind_pred, ind] += 1

                self.total_statistics['svm_mat'] = self.svm_mat
                print("You can load and save the confusion matrix from the pickled dictionary under the key 'svm_mat'")


    def knn(self, 
            CV = False,
            n_neighbors = 1,
            max_n_neighbors = 20):
        """perform KNN classification on the training and test data. This method prints progress
        updates, statistics, sets appropriate attributes for the class instance, and stores results
        and statistical data in the class instance statistics dictionary. 

        Args:
            CV (bool, optional): Toggle whether or not to do cross validation. Defaults to True.
            n_neighbors (int, optional): For CV=False, number of neighbors for single iteration of
                                            KNN. Defaults to 1.
            max_n_neighbors (int, optional): For cross validation, this is a list of number of
                                             neighbors to train KNN with for each pass of the kfold 
                                             split. Defaults to 20.
        """

        if CV:
            st = time.time()
            ks = range(1, max_n_neighbors + 1)

            ## Makes a 2-D array of to record the accuracies in
            self.knn_test_accs = np.zeros((self.n_splits, len(ks)))
            self.knn_train_accs = np.zeros((self.n_splits, len(ks)))

            i = 0
            for train_index, test_index in self.kfold.split(self.X_train, self.y_train):
                ## Keeps track of the cross validation split you are on

                print("CV Split: %s" % i)
                tt = self.X_train[train_index]
                ho = self.X_train[test_index]
                y_tt = self.y_train[train_index]
                y_ho = self.y_train[test_index]

                for neighbors in ks:
                    knn = KNeighborsClassifier(neighbors)

                    ## Fit on the tt data
                    knn.fit(tt, y_tt)
                                
                    
                    self.knn_test_accs[i] = accuracy_score(y_ho, knn.predict((ho)))
                    self.knn_train_accs[i] = accuracy_score(y_tt, knn.predict(tt))

                i = i + 1
            et = time.time() - st
            print("Elapsed Time: %s" % et)     
            
            max_index = np.argmax(np.mean(self.knn_test_accs, axis=0))

            self.neighbors_max = ks[max_index]
            print("The k with the highest AVG CV Accuracy was", 
                  "k =", self.neighbors_max)
            
            self.knn_test_acc_max = np.max(np.mean(self.knn_test_accs, axis=0))
            print("The highest CV KNN test accuracy was", self.knn_test_acc_max)

            self.knn_train_acc_max = np.mean(self.knn_train_accs[:, max_index])
            print("The corresponding CV KNN train accuracy was", self.knn_train_acc_max)

            self.total_statistics['knn_CV'] = {'test_acc': self.knn_test_accs, 
                                                   'train_acc': self.knn_train_accs, 
                                                   'test_max': self.knn_test_acc_max,
                                                   'train_max': self.knn_train_acc_max}

        else:

            ## Perform a single iteration of KNN


            knn = KNeighborsClassifier(n_neighbors)

            knn.fit(self.X_train, self.y_train)

            self.y_knn_test_pred = knn.predict(self.X_test)
            self.y_knn_train_pred = knn.predict(self.X_train)

            self.knn_test_acc = accuracy_score(self.y_test, self.y_knn_test_pred)
            print("The KNN test accuracy was", self.knn_test_acc)
            self.knn_train_acc = accuracy_score(self.y_train, self.y_knn_train_pred)
            print("The KNN train accuracy was", self.knn_train_acc)

            self.total_statistics['knn' + str(n_neighbors)] = {'test_acc': self.knn_test_acc,
                                            'train_acc': self.knn_train_acc}

            self.knn_test_pred_arr = np.concatenate((self.y_test.reshape(-1,1), self.y_knn_test_pred.reshape(-1,1)), axis = 1)

            if len(self.labels) > 0:
                self.knn_mat = np.zeros((len(self.labels) + 1, len(self.labels) + 1), dtype = int)
                self.knn_mat[0, 1:] = self.labels
                self.knn_mat[1:,0] = self.labels

                for item in self.knn_test_pred_arr:
                    test = item[0]
                    ind = np.where(self.knn_mat[0] == test)[0][0]
                    pred = item[1]
                    ind_pred = np.where(self.knn_mat[:,0] == pred)[0][0]
                    if test == pred:
                        self.knn_mat[ind, ind] += 2
                    else:
                        self.knn_mat[ind, ind] += 1
                        self.knn_mat[ind_pred, ind] += 1

                self.total_statistics['knn_mat'] = self.knn_mat
                print("You can load and save the confusion matrix from the pickled dictionary under the key 'knn_mat'")


    def log_reg_PCA(self, 
                    CV = True,
                    PCA_dim = 2,
                    PCA_dims = [2],
                    penalty = None,
                    C_val = 1.0):
        """perform PCA then logistic regression on the training and test data. This method prints progress
        updates, statistics, sets appropriate attributes for the class instance, and stores results
        and statistical data in the class instance statistics dictionary. 

        Args:
            CV (bool, optional): Toggle whether or not to do cross validation. Defaults to True.
            PCA_dim (int, optional): For CV=False, number of components on which to project before
                                     performing logistic regression. Defaults to 2.
            PCA_dims (list, optional): list of component values to project to before CV logistic
                                       regression. Defaults to [2].
        """

        if CV:
            st = time.time()

            ## Makes a 1-D array of to record the accuracies in
            self.pca_log_reg_test_accs = np.zeros((self.n_splits, len(PCA_dims)))
            self.pca_log_reg_train_accs = np.zeros((self.n_splits, len(PCA_dims)))

            i = 0
            for train_index, test_index in self.kfold.split(self.X_train, self.y_train):
                ## Keeps track of the cross validation split you are on

                print("CV Split: %s" % i)
                tt = self.X_train[train_index]
                ho = self.X_train[test_index]
                y_tt = self.y_train[train_index]
                y_ho = self.y_train[test_index]
                
                j = 0
                for n_comps in PCA_dims:

                    ## Make the PCA pipeline here
                    pca_pipe = Pipeline([('scale', StandardScaler()),
                                        ('pca', PCA(n_comps))])
                    
                    ## Fit and then get the PCA transformed tt data here
                    pca_tt = pca_pipe.fit_transform(tt)
                    
                    ## Get the transformed holdout data here
                    pca_ho = pca_pipe.transform(ho)

                    ## Logistic Regression here:
                    log_reg = LogisticRegression(max_iter=100000,
                                                 penalty=penalty,
                                                 C = C_val)
                        
                    log_reg.fit(pca_tt, y_tt)
                        
                    self.pca_log_reg_test_accs[i, j] = accuracy_score(y_ho, log_reg.predict(pca_ho))
                    self.pca_log_reg_train_accs[i, j] = accuracy_score(y_tt, log_reg.predict(pca_tt))

                    j = j + 1
                i = i + 1
            et = time.time() - st
            print("Elapsed Time: %s" % et)     
            
            max_index = np.unravel_index(np.argmax(np.mean(self.pca_log_reg_test_accs, axis=0), axis=None), 
                                       np.mean(self.pca_log_reg_test_accs, axis=0).shape)


            print("The highest AVG CV Test Accuracy corresponds to  number of components =", PCA_dims[max_index[0]])

            self.pca_log_reg_test_acc_max = np.max(np.mean(self.log_reg_test_accs, axis=0))
            print("The highest CV Logistic Regression test accuracy was", self.pca_log_reg_test_acc_max)

            self.pca_log_reg_train_acc_max = np.max(np.mean(self.log_reg_train_accs, axis = 0))
            print("The corresponding CV Logistic Regression train accuracy was", np.mean(self.pca_log_reg_train_accs[:,max_index]))

            self.total_statistics['pca_log_reg_CV'] = {'test_acc': self.pca_log_reg_test_accs, 
                                                   'train_acc': self.pca_log_reg_train_accs, 
                                                   'test_max': self.pca_log_reg_test_acc_max,
                                                   'train_max': self.pca_log_reg_train_acc_max}

        else:

            ## Perform a single iteration of PCA then log reg

            print("Projecting to dimension = ", PCA_dim)
                                ## Make the PCA pipeline here
            pca_pipe = Pipeline([('scale', StandardScaler()),
                                ('pca', PCA(PCA_dim))])
            
            pca_train = pca_pipe.fit_transform(self.X_train)

            pca_test = pca_pipe.transform(self.X_test)

            reg = LogisticRegression(max_iter=10000,
                                     penalty = penalty,
                                     C = C_val).fit(pca_train, self.y_train)

            self.y_pca_log_reg_test_pred = reg.predict(pca_test)
            self.y_pca_log_reg_train_pred = reg.predict(pca_train)

            self.pca_log_reg_test_acc = accuracy_score(self.y_test, self.y_pca_log_reg_test_pred)
            print("The Logistic Regression test accuracy was", self.pca_log_reg_test_acc)
            self.pca_log_reg_train_acc = accuracy_score(self.y_train, self.y_pca_log_reg_train_pred)
            print("The Logistic Regression train accuracy was", self.pca_log_reg_train_acc)

            self.total_statistics['pca_log_reg' + str(PCA_dim)] = {'test_acc': self.pca_log_reg_test_acc,
                                                    'train_acc': self.pca_log_reg_train_acc}
            
            self.pca_log_reg_test_pred_arr = np.concatenate((self.y_test.reshape(-1,1), self.y_pca_log_reg_test_pred.reshape(-1,1)), axis = 1)

            if len(self.labels) > 0:
                self.pca_log_reg_mat = np.zeros((len(self.labels) + 1, len(self.labels) + 1), dtype = int)
                self.pca_log_reg_mat[0, 1:] = self.labels
                self.pca_log_reg_mat[1:,0] = self.labels

                for item in self.pca_log_reg_test_pred_arr:
                    test = item[0]
                    ind = np.where(self.pca_log_reg_mat[0] == test)[0][0]
                    pred = item[1]
                    ind_pred = np.where(self.pca_log_reg_mat[:,0] == pred)[0][0]
                    if test == pred:
                        self.pca_log_reg_mat[ind, ind] += 2
                    else:
                        self.pca_log_reg_mat[ind, ind] += 1
                        self.pca_log_reg_mat[ind_pred, ind] += 1

                self.total_statistics['pca_log_reg_mat'] = self.pca_log_reg_mat
                print("You can load and save the confusion matrix from the pickled dictionary under the key 'pca_log_reg_mat'")



    def lda(self):
        """Preform Linear Discriminant Analysis
        """
        pass

    def qda(self):
        """Preform Quadratic Discriminant Analysis
        """
        pass

    def naive_bayes(self):
        """Preform Naive Bayes Classification
        """
        pass

    def rand_forest(self, 
                    max_depths_range = [i for i in range(1, 11)],
                    n_trees = [100, 250, 500]):
        """train and test a random forest ensemble classifier on the input data. This method prints progress
        updates, statistics, sets appropriate attributes for the class instance, and stores results
        and statistical data in the class instance statistics dictionary. 

        Args:
            max_depths_range (list, optional): range of max_depths to test inf the random forest. 
                                               Defaults to [i for i in range(1, 11)].
            n_trees (list, optional): list of integers corresponding to the number of classifiers to
                                      test for each of the max_depth choices. Defaults to 
                                      [100, 250, 500].
        """

        ## Make an array of zeros that will hold the cv accuracies
        self.rf_test_accs = np.zeros((self.n_splits, len(max_depths_range), len(n_trees)))
        self.rf_train_accs = np.zeros((self.n_splits, len(max_depths_range), len(n_trees)))


        i = 0
        st = time.time()
        for train_index, test_index in self.kfold.split(self.X_train, self.y_train):
            print("CV Split: %s" % i)

            tt = self.X_train[train_index]
            ho = self.X_train[test_index]
            y_tt = self.y_train[train_index]
            y_ho = self.y_train[test_index]
            
            ## Loop through the max_depth options
            j = 0
            for depth in max_depths_range:
                # print("Depth =", depth)
                ## Look through the number of estimators options
                k = 0
                for n_est in n_trees:
                    ## Make the model object, include a random state
                    rf = RandomForestClassifier(max_depth=depth,
                                                n_estimators=n_est,
                                                max_samples = int(.8*len(tt)))
                    
                    ## Fit the model
                    rf.fit(tt, y_tt)
                    
                    ## predict on the holdout set
                    pred = rf.predict(ho)
                    
                    ## Record the accuracy
                    self.rf_test_accs[i,j,k] = accuracy_score(y_ho,  pred)
                    self.rf_train_accs[i,j,k] = accuracy_score(y_tt,  rf.predict(tt))
                    k = k + 1
                j = j + 1
            i = i + 1
        
        et = time.time() - st
        print("Elapsed Time: %s" % et)  

        ## This gives you the optimal depth and number of trees
        max_index = np.unravel_index(np.argmax(np.mean(self.rf_test_accs, axis=0), axis=None), 
                                        np.mean(self.rf_test_accs, axis=0).shape)
        self.rf_opt_max_depth = max_depths_range[max_index[0]]
        self.rf_opt_n_trees = n_trees[max_index[1]]
        print("Optimal max depth =", self.rf_opt_max_depth)
        print("Optimal number of classifiers =", self.rf_opt_n_trees)

        self.rf_test_acc_max = np.mean(self.rf_test_accs, axis = 0)[max_index[0]-1, max_index[1] - 1]
        print("The CV Avg RF test accuracy was ", self.rf_test_acc_max)
        self.rf_train_acc_max = np.mean(self.rf_train_accs, axis = 0)[max_index[0]-1, max_index[1] - 1]
        print("The corresponding CV Avg RF train accuracy was ", self.rf_train_acc_max)

        self.total_statistics['rf_CV'] = {'test_acc': self.rf_test_accs, 
                                          'train_acc': self.rf_train_accs, 
                                          'test_max': self.rf_test_acc_max,
                                          'train_max': self.rf_train_acc_max}


    def save_statistics(self, filename):
        save_object_to_pickle(self.total_statistics, filename)



def save_object_to_pickle(obj, filename):
    """save any object to a .pkl binary file according to path given in filename.

    Args:
        obj (obj): Object to pickle. class instance, dict, array, etc.
        filename (str): string containing path to which to save the binary file.
    """
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_object_from_pickle(filename):
    """load an object from a binary .pkl file at the given path

    Args:
        filename (str): path to a binary .pkl file to load into a python object

    Returns:
        obj: object loaded from pickle
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj