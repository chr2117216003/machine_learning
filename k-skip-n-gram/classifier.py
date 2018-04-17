# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 20:51:20 2017

@author: Ashiqul
"""
from __future__ import division
from sklearn.model_selection import LeaveOneOut
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from Bio import SeqIO
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression


import feature_module as fm 

def jackknife(bag, tag):
    count =0
    corr =0
    x=bag
    y=tag
    accuracy = 0
    loo = LeaveOneOut()
    t= []
    p=[]
    
    for a, b in loo.split(y):
        
        x_train = x[a]; x_test = x[b]; y_train= y[a]; y_test =y [b]
        
        
        lr =LogisticRegression(C = 1, random_state=1)
    
    
        # fit the Logistic Regression model
        lr.fit(x_train, y_train)
    
        #predict the test set
        y_pred = lr.predict(x_test)
        
        #Number of misclassification
        c= (y_test != y_pred).sum()
        if c==0:
            corr= corr+1
        t.append(y_test)
        p.append(y_pred)
        
        #Show the accuracy score
        
        accuracy +=(accuracy_score(y_test, y_pred))
        # mcc+=(matthews_corrcoef(y_test, y_pred) )
        count= count+1
    
    return (corr/count)

    
    
    

def classifier(file_list, p2=10, p4=10, p5=2, p6=2, p7=5,p8=1, crossval="5"):
    
    corpus1=[]#make a blank list of corpus for the training set
    tag=[]#make the list of outcomes
    tagid = 1
    
    for files in file_list:
        fh = open(files)
       
        
        for line in SeqIO.parse(fh, "fasta"):
            line = line.seq.tostring().lower().replace("x","")
            line = line.replace('-', "")
            
           # print line
            tag.append("Class"+str(tagid))
            
            fullstring = fm.extract_motifs_pos(line, 1, p2, 1, p4, p5, p6, p7, p8)
            #fullstring = fullstring+ " "+ pos_prot_1st_word(line)
            corpus1.append(fullstring) #apperd string from each protein to corpus
        
        fh.close()
        tagid += 1
        
        
    
    
   
    #voc=['RDGYP', 'GKCMN', 'GYCYW', 'KCYCY', 'WCVWD', 'CCEHL', 'HKWCK', 'CADWA', 'WKWCV', 'LPDNV',
         #'PMH', 'WYP', 'WWKCG', 'PCFTT', 'NYPLD', 'PMKCI', 'PWC', 'NDYCL', 'LWCRY', 'MWTCD']
    corpus = np.array(corpus1) #convert corpus into numpy array
    tag = np.array(tag)  # convert tag into numpy array   
    #print corpus # print for debugging 
    #print tag # print for debugging
    del fullstring 
    del corpus1
    count = CountVectorizer(max_features=50000, vocabulary = None, max_df=0.3, min_df = 3, binary = True)#giving the CountVectorizer function a short name
    #get the vocabulary of train set to use for the test set
    
    
    
    bag = count.fit_transform(corpus) #transform the corpus(bag of words into sparse martrix)
    #print (count.vocabulary_) #count the occurence of number of words
    ##get the vocabulary of train set to use for the test set. Next time put the "voc" in
    #the vocabulary parameter of count
    voc = count.get_feature_names() 
    #print len(voc)
    bag= bag.toarray() #convert the sparsematrix into an array
    #np.place(bag, bag>0, [1])
    #print bag
    
    
    
    x = bag
    y = tag
    clfobj=   LogisticRegression(C = 1, random_state=1)
    clfobj.fit(x,y)
    #lda = LinearDiscriminantAnalysis(n_components=200)
    #lda.fit(x,y)
    #X = lda.transform(x)
    #parameterize the Logistic Regression algorithm
    if crossval=="jackknife":
        return jackknife(bag, tag), clfobj, voc
        
    else:
        crossval = int(crossval)
        cv = ShuffleSplit(n_splits=crossval, test_size=(1/crossval), random_state =1)
        clf=   LogisticRegression(C = 1, random_state=1)
        return np.mean(cross_val_score(clf, x, y, cv =cv)), clfobj, voc



def predict(f3, clfobj, voc, p2=10, p4=10, p5=2, p6=2, p7=5, p8=1):
    
    
       
    corpus2=[]#make a blank list of corpus for the training set
    
    sample = open(f3)
    ids = []
    seqs = []
    
    for line in SeqIO.parse(sample, "fasta"):
        ids.append(str(line.id).upper())
        seqs.append(str(line.seq).upper())
        line = line.seq.tostring().lower().replace("x","")
        line = line.replace('-', "")
       # print line
        
        fullstring = fm.extract_motifs_pos(line,  1, p2, 1, p4, p5, p6, p7, p8)
        #fullstring = fullstring+ " "+ pos_prot_1st_word(line)
        corpus2.append(fullstring) #apperd string from each protein to corpus
    sample.close()
    
    
    corpus = np.array(corpus2) #convert corpus into numpy array  
    #print corpus # print for debugging 
    #print tag # print for debugging
    
    count = CountVectorizer(vocabulary = voc)
    #get the vocabulary of train set to use for the test set
    
    
    
    bag_test = count.fit_transform(corpus) #transform the corpus(bag of words into sparse martrix)
    bag_test= bag_test.toarray()
    np.place(bag_test, bag_test>0, [1])
    #bag_test = pca.transform(bag_test)
    
    x_test = bag_test
    #y_test = tag
    # fit the Logistic Regression model
   
    y_pred= clfobj.predict(x_test)
    y_prob = clfobj.predict_proba(x_test)[1][1]
    
    return y_pred, ids, seqs

     
def fastaseq(f):
    ids = []
    seqs = []
         
    for lines in SeqIO.parse(f, "fasta"):
        ids.append(str(lines.id))
        seqs.append(str(lines.seq))
    return ids, seqs
    
    
    
