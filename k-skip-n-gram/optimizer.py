from feature_module import extract_motifs_pos
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from Bio import SeqIO
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression





def parameters(file_list, x1=1, x2=10, x3=1, x4=10, x5=2, x6=2, x7=5, x8=1):
    corpus1=[]#make a blank list of corpus for the training set
    tag=[]#make the list of outcomes
    tagid = 1
    for files in file_list:
        fh = open(files)
       
        fh.seek(0)  #again starts reading the file from the begining
        for line in SeqIO.parse(fh, "fasta"):
            line = line.seq.tostring().lower().replace("x","")
            line = line.replace('-', "")
            
           # print line
            tag.append("Class"+str(tagid))
            
            fullstring = extract_motifs_pos(line, 1, x2, 1, x4, x5, x6, x7, x8)
            #fullstring = fullstring+ " "+ pos_prot_1st_word(line)
            corpus1.append(fullstring) #apperd string from each protein to corpus
        
        
        tagid += 1
        fh.close()
        
    
    
       
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
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state =1)
    clf=   LogisticRegression(C = 1, random_state=1)
    return np.mean(cross_val_score(clf, x, y, cv =cv))

 

def seeding1(file_list, x_seed = 1, iteration = 0, x5=1, x6=1, x7=5, x8=1, step = 3, first_time = True):
   
    for i in range(1,26,step):
            
        #print "seed: ", i
        
        if iteration%2!=0:
            x= i
            x2 = x_seed
            
            #print "x4 is parameter"
            
        else:
            x = i
            x4 = x_seed
            #print "x2 is parameter"
            
        p = 0.0     
        
        if first_time:
            p_new= -1
            p_best = -1
        else:
            p_new = p_best
        first_time = False
        
        used = False
        
        while True:
            if iteration%2!=0:
                x4= x
            else:
                x2 = x
            
            #print "count:  ", count
            #print "value of x4: ", x4
            p = parameters(file_list, 1, x2, 1, x4, x5, x6, x7, x8)
            #print "accuracy for seed %d is: %f " % (i, p)
            #print "x4 value is" , x4
            #print "value of p", p
            #print "value of p_new", p_new
            
            if p_new < p:
                p_new = p
                #print "new value of p_new is:  ", p_new
                x+=1
                #print "yes"
                used = True
                #print used
            else:
                #print "want to see if use is true or false  :", used
                #print "no"
              
                p_best = p_new
                if used:
                    x_best = x-1
                    #print 'best accuracy for seed %d is %f' %(i, p_best)
                    #print "best parameter so far: %d" %x_best
                break 
    #print "best accuracy for seed is %f" % p_best
    #print "best parameter so far: %d" %x_best
    return x_best, p_best


def seeding2(file_list, x_seed = 1, iteration = 0, x2=1, x4=1, x7=5, x8=1, step = 1, first_time = True):
   
    for i in range(1,11,step):
            
        #print "seed: ", i
        
        if iteration%2!=0:
            x= i
            x5 = x_seed
            
            #print "x4 is parameter"
            
        else:
            x = i
            x6= x_seed
            #print "x2 is parameter"
            
        p = 0.0     
        
        if first_time:
            p_new= -1
            p_best = -1
        else:
            p_new = p_best
        first_time = False
        
        used = False
        
        while True:
            if iteration%2!=0:
                x6= x
            else:
                x5 = x
            
            #print "count:  ", count
            #print "value of x4: ", x4
            p = parameters(file_list, 0, x2, 0, x4, x5, x6, x7, x8)
            #print "accuracy for seed %d is: %f " % (i, p)
            #print "x4 value is" , x4
            #print "value of p", p
            #print "value of p_new", p_new
            
            if p_new < p:
                p_new = p
                #print "new value of p_new is:  ", p_new
                x+=1
                #print "yes"
                used = True
                #print used
            else:
                #print "want to see if use is true or false  :", used
                #print "no"
              
                p_best = p_new
                if used:
                    x_best = x-1
                    #print 'best accuracy for seed %d is %f' %(i, p_best)
                    #print "best parameter so far: %d" %x_best
                break 
    #print "best accuracy for seed is %f" % p_best
    #print "best parameter so far: %d" %x_best
    return x_best, p_best


def seeding3(file_list, x_seed = 1, iteration = 0, x2=1, x4=1, x5=1, x6=1, step = 1, first_time = True):
   
    for i in range(1,11,step):
            
        #print "seed: ", i
        
        if iteration%2!=0:
            x= i
            x7 = x_seed
            
            #print "x4 is parameter"
            
        else:
            x = i
            x8= x_seed
            #print "x2 is parameter"
            
        p = 0.0     
        
        if first_time:
            p_new= -1
            p_best = -1
        else:
            p_new = p_best
        first_time = False
        
        used = False
        
        while True:
            if iteration%2!=0:
                x8= x
            else:
                x7 = x
            
            #print "count:  ", count
            #print "value of x4: ", x4
            p = parameters(file_list, 0, x2, 0, x4, x5, x6, x7, x8)
            #print "accuracy for seed %d is: %f " % (i, p)
            #print "x4 value is" , x4
            #print "value of p", p
            #print "value of p_new", p_new
            
            if p_new < p:
                p_new = p
                #print "new value of p_new is:  ", p_new
                x+=1
                #print "yes"
                used = True
                #print used
            else:
                #print "want to see if use is true or false  :", used
                #print "no"
              
                p_best = p_new
                if used:
                    x_best = x-1
                    #print 'best accuracy for seed %d is %f' %(i, p_best)
                    #print "best parameter so far: %d" %x_best
                break 
    #print "best accuracy for seed is %f" % p_best
    #print "best parameter so far: %d" %x_best
    return x_best, p_best


def sub_tuning1(file_list,seed, x5=1, x6=1, x7=5, x8=1):
    x5=1; x6=1; x7=1;
    initial = True
    iteration = 0
    while True:
        
        if initial:        
            best_accuracy = -1
            p_best = -1
            x = seed
        else:
            if best_accuracy < p_best:
                best_accuracy = p_best
                x = x_best 
            else:
                
                if iteration%2==0:
                    x4_best= x
                    print "x4 best so far:", x4_best
                else:
                    x2_best = x
                    print "x2 best so far:", x2_best
                print "Best accuracy %f" % best_accuracy
                return x2_best, x4_best
                break
        iteration += 1 
        initial = False        
        x_best, p_best = seeding1(file_list, x, iteration)
        if iteration%2!=0:
            x4_best= x_best
            print "x4 best so far:", x4_best
        else:
            x2_best = x_best
            print "x2 best so far:", x2_best
            



def sub_tuning2(file_list, x2=1, x4=1, x7=5, x8=1):
    initial = True
    iteration = 0
    while True:
        
        if initial:        
            best_accuracy = -1
            p_best = -1
            x = 1
        else:
            if best_accuracy < p_best:
                best_accuracy = p_best
                x = x_best 
            else:
                print "Best accuracy  %f" % best_accuracy
                return x5_best, x6_best
                break
        iteration += 1 
        initial = False        
        x_best, p_best = seeding2(file_list, x, iteration, x2, x4)
        if iteration%2!=0:
            x6_best= x_best
            print "x6 best so far:", x6_best
        else:
            x5_best = x_best
            print "x5 best so far:", x5_best
    print "Best accuracy  %f" % best_accuracy

def sub_tuning3(file_list, x2=1, x4=1, x5=1, x6=1):
    
    initial = True
    iteration = 0
    while True:
        
        if initial:        
            best_accuracy = -1
            p_best = -1
            x = 1
        else:
            if best_accuracy < p_best:
                best_accuracy = p_best
                x = x_best 
            else:
                print "Best accuracy  %f" % best_accuracy
                return x7_best, x8_best, best_accuracy
                break
        iteration += 1 
        initial = False        
        x_best, p_best = seeding3(file_list, x, iteration, x2, x4, x5, x6)
        if iteration%2!=0:
            x8_best= x_best
            print "x8 best so far:", x8_best
        else:
            x7_best = x_best
            print "x7 best so far:", x7_best
            


def tune(file_list, seed):
    x2, x4 = sub_tuning1(file_list, seed)
    x5, x6 = sub_tuning2(file_list,  x2, x4)
    x7, x8, best_accuracy = sub_tuning3(file_list, x2,x4, x5, x6)
    lst = [x2, x4, x5, x6, x7, x8, best_accuracy] 
    return lst
    

def optimization(file_list):
    
    best = [0, 0, 0, 0, 0, 0, 0]
    for i in (range(1, 3, 2)):
        f = str(tune(file_list, i))
        if best[6]< f[6]:
            best = f
            print best
        
        #fh.write(str(i));fh.write(f);fh.write("\n") 
        print f
    return best 
    

