from sklearn.preprocessing import RobustScaler,StandardScaler,QuantileTransformer
import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, roc_auc_score
from sklearn.model_selection import  KFold, train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import glob
import random
class Wavelet_Classifier():
    def __init__(self):
        self.evaluationName = ['Precision', 'F1Score', 'Accuracy', 'Recall', 'Matt', 'Auc']

    def Obtain_train_Data(self,path):
        temp_list_to_save_SVM = []
        y = []
        for line in open(path, 'r'):
            list1 = line.rstrip().split()
            if len(list1) == 2:
                y.append(list1[1])
                continue
            temp_list_to_save_SVM.append([float(i) for i in list1])
        return temp_list_to_save_SVM, y

    #This function calculates Auc measrement. y_test (real label)  and y_pred ( predicted label) are two inputed parameters

    def multiclass_roc_auc_score(self, y_test, y_pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_test, y_pred, average=average)

    #First it accpets xTrain, yTrain as training dataset and xTest,yTest as test dataset. the classifier as machine learning model define as input parameter
    # as well

    def doClassifyTrainAndTest(self, xTrain,yTrain,xTest,yTest, classifier):
        evlP = np.zeros(6)
        classifier.fit(xTrain, yTrain)
        y_pred = classifier.predict(xTest)
        y_test =yTest

        evlP[0] = (precision_score(y_test, y_pred, average='micro'))
        evlP[1] = (f1_score(y_test, y_pred, average='macro'))
        evlP[2] = (accuracy_score(y_test, y_pred))
        evlP[3] = (recall_score(y_test, y_pred, average="weighted"))
        evlP[4] = (matthews_corrcoef(y_test, y_pred))
        evlP[5] = self.multiclass_roc_auc_score(y_test, y_pred)

        return evlP

    #this function get xTrain, yTarin, xTest,yTest as input parameters, then
    #make 8 classifier object and apply indpendent test on them

    def applyAllmodelTrainAndTest(self, xTrain,yTrain,xTest,yTest):
        ada = AdaBoostClassifier(n_estimators=200, base_estimator=None, learning_rate=0.1, random_state=1)
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        nivebase = GaussianNB()
        dt = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
        lr = LogisticRegression(random_state=0)
        svclassifier = SVC(kernel='rbf', degree=3, C=2)
        randomforest = RandomForestClassifier(n_estimators=200)  # Train the model on training data
        mlp = MLPClassifier(hidden_layer_sizes=(20, 10, 5), max_iter=1000, alpha=1e-4, solver='sgd', verbose=0,
                            tol=1e-4, random_state=1, learning_rate_init=.08)

        paramada = self.doClassifyTrainAndTest(xTrain,yTrain,xTest,yTest, ada)
        paramknn = self.doClassifyTrainAndTest(xTrain,yTrain,xTest,yTest, knn)
        paramnb = self.doClassifyTrainAndTest(xTrain,yTrain,xTest,yTest, nivebase)
        paramdt = self.doClassifyTrainAndTest(xTrain,yTrain,xTest,yTest, dt)
        paramlr = self.doClassifyTrainAndTest(xTrain,yTrain,xTest,yTest, lr)
        paramsvm = self.doClassifyTrainAndTest(xTrain,yTrain,xTest,yTest, svclassifier)
        paramrf = self.doClassifyTrainAndTest(xTrain,yTrain,xTest,yTest, randomforest)
        parammlp = self.doClassifyTrainAndTest(xTrain,yTrain,xTest,yTest, mlp)
        result = np.zeros((8,6))
        result[0,:] = paramada
        result[1,:] = paramknn
        result[2,:] = paramnb
        result[3,:] = paramdt
        result[4,:] = paramlr
        result[5,:] = paramsvm
        result[6,:] = paramrf
        result[7,:] = parammlp
        return result

    #this function get X as feature vector, y as proteins label, classifier as machine learning model and kf as KFold object input parameters, then
    #divide X into nfold (default is 5) parts,so that each part is contain testset and trainset for evaluation.
    #  F1 , accuracy, precision,recll are the main performace evaluators used in this function.
    #Finally the mean of nfold part would returned as the results

    def doClassifyCrossValidation(self, X, y, classifier, kf,nfold=5):
        Data = X
        evlP = [[0 for x in range(6)] for YY in range(nfold)]
        k = 0
        for train_index, test_index in kf.split(Data):
            classifier.fit(Data[train_index], y[train_index])
            y_pred = classifier.predict(Data[test_index])
            y_test = y[test_index]

            evlP[k][0] = (precision_score(y_test, y_pred, average='micro'))
            evlP[k][1] = (f1_score(y_test, y_pred, average='macro'))
            evlP[k][2] = (accuracy_score(y_test, y_pred))
            evlP[k][3] = (recall_score(y_test, y_pred, average="weighted"))
            evlP[k][4] = (matthews_corrcoef(y_test, y_pred))
            evlP[k][5] = self.multiclass_roc_auc_score(y_test, y_pred)
            k += 1

        average = np.matrix(evlP)
        average = average.mean(axis=0)
        return average[0]

    #applyAllModelCrossValidation would accepted X,Y as protein features and protein labels respectively
    #and make 8 classifer (ada,knn, niavebase, dt,le, svcclassifier, randomforest and mlp), then apply 5 fold cross validation
    #for each of them.
    #the output of this function is joined performance of all classifiers as a unit matrix

    def applyAllmodelCrossValidation(self, X, Y):
        ada = AdaBoostClassifier(n_estimators=200, base_estimator=None, learning_rate=0.1, random_state=1)
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        nivebase = GaussianNB()
        dt = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
        lr = LogisticRegression(random_state=0)
        svclassifier = SVC(kernel='rbf', degree=3, C=2)
        randomforest = RandomForestClassifier(n_estimators=200)  # Train the model on training data
        mlp = MLPClassifier(hidden_layer_sizes=(20, 10, 5), max_iter=1000, alpha=1e-4, solver='sgd', verbose=0,
                            tol=1e-4, random_state=1, learning_rate_init=.08)

        kf = KFold(n_splits=5, shuffle=True, random_state=random.randint(1,100))

        paramada = self.doClassifyCrossValidation(X, Y, ada,kf)
        paramknn = self.doClassifyCrossValidation(X, Y, knn,kf)
        paramnb = self.doClassifyCrossValidation(X, Y, nivebase,kf)
        paramdt = self.doClassifyCrossValidation(X, Y, dt,kf)
        paramlr = self.doClassifyCrossValidation(X, Y, lr,kf)
        paramsvm = self.doClassifyCrossValidation(X, Y, svclassifier,kf)
        paramrf = self.doClassifyCrossValidation(X, Y, randomforest,kf)
        parammlp = self.doClassifyCrossValidation(X, Y, mlp,kf)
        result = np.zeros((8,6))
        result[0,:] = paramada
        result[1,:] = paramknn
        result[2,:] = paramnb
        result[3,:] = paramdt
        result[4,:] = paramlr
        result[5,:] = paramsvm
        result[6,:] = paramrf
        result[7,:] = parammlp
        return result

    #Use this function to remove thoses columns which have correlated more than threshold, the default value for threshold is 0.7
    # you could chande threshold in line 166
    #dataframe as an input parameter is a pandas DataFrame containing all extracted wavelet vectors


    def RemoveCorrelated(self, dataframe):
        corr_matrix = dataframe.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = {}
        for i in range(upper.values.shape[0]):
            for j in range(i + 1, upper.values.shape[0]):
                if upper.values[i, j] >= 0.70:
                    to_drop[upper.columns[j]] = 1

        uncorrelated_data = dataframe.drop(to_drop.keys(), axis=1)
        return uncorrelated_data



    #By using this function the main process of 100* 5-fold cross validation runs, path parameter is the
    #path of the directoary containing feature_vector files (please refer instruction for the name of the file and more information).
    def DoClassify(self,path):

        run_times = 5
        filenames = os.path.basename(path)
        performance_cross_ada = np.zeros((run_times, 6))
        performance_cross_knn = np.zeros((run_times, 6))
        performance_cross_nb = np.zeros((run_times, 6))
        performance_cross_dt = np.zeros((run_times, 6))
        performance_cross_lr = np.zeros((run_times, 6))
        performance_cross_svm = np.zeros((run_times, 6))
        performance_cross_rf = np.zeros((run_times, 6))
        performance_cross_mlp = np.zeros((run_times, 6))

        performance_jack_ada = np.zeros((run_times, 6))
        performance_jack_knn = np.zeros((run_times, 6))
        performance_jack_nb = np.zeros((run_times, 6))
        performance_jack_dt = np.zeros((run_times, 6))
        performance_jack_lr = np.zeros((run_times, 6))
        performance_jack_svm = np.zeros((run_times, 6))
        performance_jack_rf = np.zeros((run_times, 6))
        performance_jack_mlp = np.zeros((run_times, 6))

        X, y = self.Obtain_train_Data(path)
        X = RobustScaler().fit_transform(X)
        X = np.array(X)
        y = np.array(y)
        indices = np.arange(X.shape[0])
        #run_times parameter indicates the number of  5 fold cross validation's run
        for runs in range(run_times):

            print (runs)
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, test_size=0.2,random_state=random.randint(1,100))
            X_train = X[idx_train,:]
            y_train = y[idx_train]
            X_test= X[idx_test,:]
            y_test= y[idx_test]

            result = self.applyAllmodelCrossValidation(X_train,y_train)

            performance_cross_ada[runs, :] = result[0,:]
            performance_cross_knn[runs, :] = result[1,:]
            performance_cross_nb[runs, :] = result[2,:]
            performance_cross_dt[runs, :] = result[3,:]
            performance_cross_lr[runs, :] = result[4,:]
            performance_cross_svm[runs, :] = result[5,:]
            performance_cross_rf[runs, :] = result[6,:]
            performance_cross_mlp[runs, :] = result[7,:]

            result = self.applyAllmodelTrainAndTest(X_train, y_train,X_test,y_test)

            performance_jack_ada[runs, :] = result[0,:]
            performance_jack_knn[runs, :] = result[1,:]
            performance_jack_nb[runs, :] = result[2,:]
            performance_jack_dt[runs, :] = result[3,:]
            performance_jack_lr[runs, :] = result[4,:]
            performance_jack_svm[runs, :] = result[5,:]
            performance_jack_rf[runs, :] = result[6,:]
            performance_jack_mlp[runs, :] = result[7,:]

        evaluationName = ['Precision', 'F1Score', 'Accuracy', 'Recall', 'Matt', 'Auc']

        ada_mean, ada_var = self.Mean_Variance(performance_cross_ada)
        knn_mean, knn_var = self.Mean_Variance(performance_cross_knn)
        nb_mean, nb_var = self.Mean_Variance(performance_cross_nb)
        dt_mean, dt_var = self.Mean_Variance(performance_cross_dt)
        lr_mean, lr_var = self.Mean_Variance(performance_cross_lr)
        svm_mean, svm_var = self.Mean_Variance(performance_cross_svm)
        rf_mean, rf_var = self.Mean_Variance(performance_cross_rf)
        mlp_mean, mlp_var = self.Mean_Variance(performance_cross_mlp)

        df1 = pd.DataFrame({'Ada': ada_mean,
                            'KNN': knn_mean,
                            'NB': nb_mean,
                            'DT': dt_mean,
                            'LR': lr_mean,
                            'SVM': svm_mean,
                            'RF': rf_mean,
                            'MLP': mlp_mean}, index=evaluationName)
        df2 = pd.DataFrame({'Ada': ada_var,
                            'KNN': knn_var,
                            'NB': nb_var,
                            'DT': dt_var,
                            'LR': lr_var,
                            'SVM': svm_var,
                            'RF': rf_var,
                            'MLP': mlp_var}, index=evaluationName)
        with pd.ExcelWriter('Result/{}_crossvalidation.xlsx'.format(filenames)) as writer:
            df1.to_excel(writer, sheet_name='Sheet1')
            df2.to_excel(writer, sheet_name='Sheet2')

        ada_mean, ada_var = self.Mean_Variance(performance_jack_ada)
        knn_mean, knn_var = self.Mean_Variance(performance_jack_knn)
        nb_mean, nb_var = self.Mean_Variance(performance_jack_nb)
        dt_mean, dt_var = self.Mean_Variance(performance_jack_dt)
        lr_mean, lr_var = self.Mean_Variance(performance_jack_lr)
        svm_mean, svm_var = self.Mean_Variance(performance_jack_svm)
        rf_mean, rf_var = self.Mean_Variance(performance_jack_rf)
        mlp_mean, mlp_var = self.Mean_Variance(performance_jack_mlp)

        df1 = pd.DataFrame({'Ada': ada_mean,
                            'KNN': knn_mean,
                            'NB': nb_mean,
                            'DT': dt_mean,
                            'LR': lr_mean,
                            'SVM': svm_mean,
                            'RF': rf_mean,
                            'MLP': mlp_mean}, index=evaluationName)
        df2 = pd.DataFrame({'Ada': ada_var,
                            'KNN': knn_var,
                            'NB': nb_var,
                            'DT': dt_var,
                            'LR': lr_var,
                            'SVM': svm_var,
                            'RF': rf_var,
                            'MLP': mlp_var}, index=evaluationName)
        with pd.ExcelWriter('Result/{}_jacktest.xlsx'.format(filenames)) as writer:
            df1.to_excel(writer, sheet_name='Sheet1')
            df2.to_excel(writer, sheet_name='Sheet2')
        return rf_mean[0]

    #This function accept a matrix as an input and calulate the average and variance of each column seperatly and retuned them.

    def Mean_Variance(self,arr):
        mean_ada = np.mean(arr, axis=0)
        var_ada = np.var(arr, axis=0)
        return mean_ada, var_ada
bestmodel = 0
bestper = 0

#feature_xlsx_path is the path of directory, includes filter bank feature vector text files

feature_xlsx_path = "Filter_banks_feature_vectors/*.txt"
for item in glob.glob(feature_xlsx_path):
    print (item)
    moonlight_object = Wavelet_Classifier()
    perfomance = moonlight_object.DoClassify(item)
    if (perfomance > bestper):
        bestper = perfomance
        bestmodel = item
print (bestmodel)
print (bestper)



