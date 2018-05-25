#!/usr/bin/env python
# encoding:utf-8
import os
import sys
import getopt
import threading
import pandas as pd
import math
import numpy as np
from time import clock
from sklearn.feature_selection import  f_classif
from sklearn.externals.joblib import Memory
from sklearn import  metrics
import easy_excel

import itertools
from sklearn.model_selection import KFold  
from sklearn import svm
from sklearn.model_selection import train_test_split
import math
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import easy_excel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import *
import sklearn.ensemble
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB  
import subprocess
from sklearn.utils import shuffle
import itertools
from sklearn.ensemble import GradientBoostingClassifier
import sys
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import load_svmlight_file
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier

def performance(labelArr, predictArr):
    #labelArr[i] is actual value,predictArr[i] is predict value
    TP = 0.; TN = 0.; FP = 0.; FN = 0.
    for i in range(len(labelArr)):
        if labelArr[i] == 1 and predictArr[i] == 1:
            TP += 1.
        if labelArr[i] == 1 and predictArr[i] == 0:
            FN += 1.
        if labelArr[i] == 0 and predictArr[i] == 1:
            FP += 1.
        if labelArr[i] == 0 and predictArr[i] == 0:
            TN += 1.
    if (TP + FN)==0:
        SN=0
    else:
        SN = TP/(TP + FN) #Sensitivity = TP/P  and P = TP + FN
    if (FP+TN)==0:
        SP=0
    else:
        SP = TN/(FP + TN) #Specificity = TN/N  and N = TN + FP
    if (TP+FP)==0:
        precision=0
    else:
        precision=TP/(TP+FP)
    if (TP+FN)==0:
        recall=0
    else:
        recall=TP/(TP+FN)
    GM=math.sqrt(recall*SP)
    #MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return precision,recall,SN,SP,GM,TP,TN,FP,FN


mem = Memory("./mycache")
@mem.cache
def get_data(file_name):
    data = load_svmlight_file(file_name)
    return data[0], data[1]


def csv_and_arff2svm(arff_files):
    svm_files = []
    for arff_file in arff_files:
        name = arff_file[0: arff_file.rindex('.')]
        tpe = arff_file[arff_file.rindex('.')+1:]
        svm_file = name+".libsvm"
        svm_files.append(svm_file)
        if tpe == "arff":
            if os.path.exists(svm_file):
                pass
            else:
                f = open(arff_file)
                w = open(svm_file, 'w')
                flag = False
                for line in f.readlines():
                    if flag:
                        if line.strip() == '':
                            continue
                        temp = line.strip('\n').strip('\r').split(',')
                        w.write(temp[len(temp)-1])
                        for i in range(len(temp)-1):
                            w.write(' '+str(i+1)+':'+str(temp[i]))
                        w.write('\n')
                    else:
                        line = line.upper()
                        if line.startswith('@DATA') or flag:
                            flag = True
                f.close()
                w.close()
        elif tpe == "csv":
            if os.path.exists(svm_file):
                pass
            else:
                f = open(arff_file)
                w = open(svm_file, 'w')
                for line in f.readlines():
                    if line.strip() == '':
                        continue
                    temp = line.strip('\n').strip('\r').split(',')
                    w.write(temp[len(temp)-1])
                    for i in range(len(temp)-1):
                        w.write(' '+str(i+1)+':'+str(temp[i]))
                    w.write('\n')
                f.close()
                w.close()
        elif tpe == "libsvm":
            continue
        else:
            print "File format error! Arff and libsvm are passed."
            sys.exit()
    return svm_files
opts, args = getopt.getopt(sys.argv[1:], "hi:c:t:o:s:m:", )
for op, value in opts:
    if op == "-i":
        input_files = str(value)
        input_files = input_files.replace(" ", "").split(',')
        for input_file in input_files:
            if input_file == "":
                print "Warning: please insure no blank in your input files !"
                sys.exit()
    elif op == "-c":
        cv = int(value)
    elif op == "-t":
        split_rate = float(value)
    elif op == "-o":
        excel_name = str(value)
    elif op == "-s":
        if str(value) != "0":
            isSearch = True
    elif op == "-m":
        if str(value) == "0":
            isMultipleThread = False
    elif op == "-h":
        print 'Cross-Validate: python easy_classify.py -i {input_file.libsvm} -c {int: cross validate folds}'
        print 'Train-Test: python easy_classify.py -i {input_file.libsvm} -t {float: test size rate of file}'
        print 'More information: https://github.com/ShixiangWan/Easy-Classify'
        sys.exit()
if __name__ =="__main__":
    path=""
    outputname="svm_f-score"
    name=outputname
    print '*** Validating file format ...'
    input_files = csv_and_arff2svm(input_files)
    for input_file in input_files:
        # 导入原始数据
        X, Y = get_data(input_file)
        train_data = X.todense()
        train_data=np.array(train_data)
        F, pval = f_classif(train_data, Y)
        idx = np.argsort(F)
        selected_list_=idx[::-1]
        F_sort_value=[F[e] for e in selected_list_]
        with open("all_dimension_results.txt",'a') as f:
                f.write(str(F_sort_value)+"\n")
        print F_sort_value
        with open("all_dimension_results.txt",'a') as f:
                f.write(str(selected_list_)+"\n")
        print selected_list_
        bestACC=0
        bestC=0
        bestgamma=0
        best_dimension=0
        all_dimension_results=[]
        for select_num in xrange(1,len(train_data[0])+1):
            train_data2=train_data
            print np.array(train_data).shape
            print np.array(train_data2).shape
            selected_list_2=selected_list_[xrange(select_num)]
            X_train=pd.DataFrame(train_data2)
            X_train=X_train.iloc[:,selected_list_2]
            X = np.array(X_train)
            svc = svm.SVC()
            parameters = {'kernel': ['rbf'], 'C':map(lambda x:2**x,np.linspace(-2,5,7)), 'gamma':map(lambda x:2**x,np.linspace(-5,2,7))}
            clf = GridSearchCV(svc, parameters, cv=cv, n_jobs=2, scoring='accuracy')
            clf.fit(X, Y)
            C=clf.best_params_['C']
            joblib.dump(clf,path+outputname+str(select_num)+".model")
            gamma=clf.best_params_['gamma']
            y_predict=cross_val_predict(svm.SVC(kernel='rbf',C=C,gamma=gamma),X,Y,cv=cv,n_jobs=2)
            y_predict_prob=cross_val_predict(svm.SVC(kernel='rbf',C=C,gamma=gamma,probability=True),X,Y,cv=cv,n_jobs=2,method='predict_proba')
            predict_save=[Y.astype(int),y_predict.astype(int),y_predict_prob[:,1]]
            predict_save=np.array(predict_save).T
            pd.DataFrame(predict_save).to_csv(path+outputname+"_"+'_predict_crossvalidation.csv',header=None,index=False)
            ROC_AUC_area=metrics.roc_auc_score(Y,y_predict)
            ACC=metrics.accuracy_score(Y,y_predict)
            precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(Y, y_predict)
            F1_Score=metrics.f1_score(Y, y_predict)
            F_measure=F1_Score
            MCC=metrics.matthews_corrcoef(Y, y_predict)
            pos=TP+FN
            neg=FP+TN
            savedata=[[['svm'+"C:"+str(C)+"gamma:"+str(gamma),ACC,precision, recall,SN, SP, GM,F_measure,F1_Score,MCC,ROC_AUC_area,TP,FN,FP,TN,pos,neg]]]
            if ACC>bestACC:
                bestACC=ACC
                bestgamma=gamma
                bestC=C
                best_dimension=X.shape[1]
            print savedata
            print X.shape[1]
            with open("all_dimension_results.txt",'a') as f:
                f.write(str(savedata)+"\n")
            all_dimension_results.append(savedata)
        print bestACC
        print bestC
        print bestgamma
        print best_dimension
        easy_excel.save("svm_crossvalidation",[str(X.shape[1])],savedata,path+'cross_validation_'+name+'.xls')