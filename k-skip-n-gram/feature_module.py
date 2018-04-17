# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 21:01:36 2016

@author: Ashiqul
"""
from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 22:11:28 2016

@author: Ashiqul
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multioutput import MultiOutputClassifier

def buffer_length(l, n):
    while l%n!=0:
        l+=1
    return l
#buffer_length(27)==30
#This function will substitute each protein residues with x except the first and the last one" 
def two_aa_relation(x, dis_buffer=False):
    """ 
This function will take one string argument and return a modified string variable
protseq = "aytg"
result = two_aa_relation(protseq)
print result
axxg
""" 
    
    x=list(x)
    if dis_buffer==False:
        x[1:-1]='x'*(len(x)-2)
        return (''.join(x))        
    else:
        dis = buffer_length((len(x)-2), dis_buffer)
        x[1:-1]='x'*dis
        return (''.join(x))

#Function to extract all posible length of words from all sliding window of 
#of a protein sequence
def prot_words(seq, max_loop=20):
    """
This function will take a protein string as an argument and return a list.
the index of the list indicates number of letters in sliding window. 
protseq = 'astc' 
result = prot_words(protseq)
print result
#['', 'xa xs xt xc', 'as st tc', 'ast stc', 'astc']
print result[2]
"""  
    
    line = ['']
    line.append(('x'+" x".join(seq)))
    if max_loop>len(seq):
        max_loop=len(seq)
    l = 1
    while l < max_loop:
        line.append( " ".join(seq[i:i+(l+1)] for i in range(0, len(seq)-l, 1))) #make word with every 2 letter
        l+=1
    return line
"""
This function will take a protein string as an argument and return a list.
the index of the list indicates number of letters in sliding window. 
protseq = 'astcrta' 
result = pos_prot_words(protseq, 5, 1)
print result
#['', 'xa xs xt xc', 'as st tc', 'ast stc', 'astc']
print result[2]
"""  
def pos_prot_words(seq, max_loop=20,  pos_length= -1, pos_buffer=5,):
    """
    This function will take a protein string as an argument and return a two lists and one value.
    the index of the list indicates number of letters in sliding window. if there is any positive
    value of pos_length argument, then that the list will give a buffered positional value the 
    subtrings that has a same length as the pos_lenth argument. 
    protseq = 'astc' 
    result = pos_prot_words(protseq, 20, 1)
    print result
    #['', 'xa xs xt xc', 'as st tc', 'ast stc', 'astc']
    print result[2]
    """  
    
    buf= pos_buffer # will help to buffer position based on motif length
    length = len(seq)+1 #to avoid the last amino acid with a zero length
    line = [''] #initialize a list with an empty component 
    line2= [''] #initialize another list with an empty component
    if max_loop>len(seq):#to increase the efficiency of the function, only upto motif with the required maximum letter will be extracted. 
        max_loop=len(seq)#if the required biggest motif is bigger than protein sequence, then the maximum motif will be automatically assigned as the protein length. 
    l = 0 #start from the single letter word
    while l < max_loop: #number of letter in a word cannot exit the length of of the protein
        x=range(0, len(seq)-l, 1)#makeing a iterator to iterate over the sequence
        y= range(1, len(seq)+1) #makeing a iterator to find the onset of a motif. That will help to measure the distance from the onset of the motif to the C-terminus end of the protein. By this way, we can calculate the possinal value from the C-terminus of the protein.
        # use tuple with list comprehension to make compact code
        words= str()#initialize a black string do to accumulate words for a list of words with same number of letter. 
        words2=str()#initialize a black string do to accumulate words for a list of words with same number of letter. 
        for i, j in zip(x,y):#Want loop though two operators at the same time
            if l==0:
                #print i, j
                word2 = seq[i]
                word  = seq[i]+str(buffer_length((length-j),(buf+l))).zfill(5)
                #print word
            word2=seq[i:(i+l+1)]#assaigning word2
            word = seq[i:(i+l+1)]+str(buffer_length((length-j),(buf+l))).zfill(5) #assaigning word with positional value. (length-j) gives the distance from the onset of the motif.(buf +1) helps to increase distance buffering with increase of motif lenghts. zfill gives the position by five digits. if a onset position is 10 from the C-terminus of a protein, then it will be showed in a five digit like 00010. this kind of presentation will be useful to take out the position from the words.
            possition = -5 #Ready to take out the five digit positional identity
            if l==0: #in case of single letter word. We will add a 'x' infront of the residue to avoid the 'stop word' problem in text mining. 
                word="x"+word #this word will be used for positional indentity
                word2="x"+word2#this word2 will be used for the argument of pos_dist_relation
            if pos_length>=(l+1):# when the pos_length parameter is equal of greater that the protein number of letters in a word(motif), that will give the positional value. 
                possition = len(word) #Possition index will become the whole length of the word.
            words= words+ word[0:possition]+ " "# words will be accumulated using space
            words2=words2+word2+" "#same thing like above is going on.
        words = words.rstrip()#the last space the string accumulated with words will be stripped.
        word2=word2.rstrip()#same thing like above is going on. 
        line.append(words)#this list will be used to get the motif
        line2.append(words2)#this list will be used as the parameter of relationship argument in pos_dist_relation function
        l+=1#accumulator for the while loop
    return line2, length, line #the first two values will be used in pos_dist_relation function. 
    
#Function to replace the residue of sequences of all possible length of the sliding windows 
#of a protein with x except the first and last one  
def pos_dist_relation(relations, length=0, d_pos_length=-1, pos_buffer = 5, dis_buffer = False):
    """    
    This function will take a list as argument and return a list as 
    a result indicating the distance based relationship bettween two amino acids
    in a protein. The index a returned list will indicate the size of words
    with the x letters. The argurments relations and length come as outout of pos_prot_words
    function. d_pos_length argument provides the position value of a distance motif
    with a size corresponding to the value of this argument. dis_buffer argument buffers
    the length of a distance motif.     
    
    """    
    buf = pos_buffer #the position will be buffered for every 5 residues. 
    lines = ['', ''] #The initial list will have two black strings
    l= 3 # the index of each list in the following loop will start from 3. Because the first index will be blank, and second index will be one letter word. So there should not be any distance for one letter words. 
    while l<len(relations):#the loop will continue until it reach to the same length of the given list.
        #print relations[l]        
        line=relations[l].split()#this line will split the string by blank space. Each string is a component of the given list. 
        #print line #to trouble shoot
        words=str() #intialize a blank string where words of from each string will deposited
        y= range(1, (length-1))#Making an iterable list which will range from 1 to length 
        for i, j in zip(line,y):#Want loop though two operators at the same time
            #print i, j
            word = two_aa_relation(i, dis_buffer)+str(buffer_length((length-j),(buf+(l-1)))).zfill(5)#assaigning word with positional value. (length-j) gives the distance from the onset of the motif.(buf +1) helps to increase distance buffering with increase of motif lenghts. zfill gives the position by five digits. if a onset position is 10 from the C-terminus of a protein, then it will be showed in a five digit like 00010. this kind of presentation will be useful to take out the position from the words.
            #above, the dis_buffer parameter will work for dis_buffer argumetn in two_aa_relation function. If this parameter is not false, that will buffer the distance according to its value. if the value is 3, the distance will increase for every 3 distance relationship. ie. a1v, a2v, a3v all will be a3v or axxxv. a4v will be axxxxxxv.          
            possition = -5  #Ready to take out the five digit positional identity          
            if d_pos_length>=(l+1):# when the d_pos_length parameter is equal of greater that the protein number of letters in a word(motif), that will give the positional value. 
                possition = len(word)    #Possition index will become the whole length of the word.       
            words= words+word[0:possition]+ " "# words will be accumulated using space
        words = words.rstrip()#the last space the string accumulated with words will be stripped.
        lines.append(words)#this list will be used to get the motif
        l+=1#accumulator for the while loop
    return lines
    



   
def extract_motifs_pos(seq = "    ", minwl=1, maxwl=1, mindl=1, maxdl=1, pos_length=-1, d_pos_length=-1, pos_buffer = 5, dis_buffer = False):
    """
    extract_motifs function will extract motifs from a protein sequence using two functions:
    prot_words and dist_words. 
    
    This function will take five paratmeters:
    
    seq:- a string type variable. this should the protein sequence. default value of 
          seq is "    "(4 spaces).
    minwl:- an interger not less than 1 and more than length of the protein. 
            this integer refers the mininum length of a word generated from sliding 
            windows. default value of minwl is 1.
    maxwl:- an interger not less than 1 and more than length of the protein.
            this integer refers the maximun length of a word generated from sliding 
            windows. default value of maxwl is 1. 
    mindl:- an interger not less than 1 and should be two less than length of the 
            protein. this integer refers the minimum length of a distance relation between 
            two amino acids.default value of minwl is 1.
    maxdl:- an interger not less than 1 and should be two less than length of the 
            protein.this integer refers the maximun length of a distance relation between 
            two amino acids. default value of maxwl is 1. 
    pos_length:- an integer. if there is any positive
                 value of pos_length argument, then that the list will give a buffered 
                 positional value the subtrings that has a same length as the pos_lenth 
                 argument. 
    d_pos_length: an integer.  d_pos_lenght argument provides the position value of a 
                  distance motif with a size corresponding to the value of this argument.
    pos_buffer: an interger. This argument buffer the positional values of the motifs.
    dis_buffer: an integer.dis_buffer argument buffers the length of a distance motif. 
    example:
        proteinseq = "emrtyqwcv"
        result = extract_motifs_pos(proteinseq, 1, 10, 1, 1, 1, 0)
        print result
        output = 'xe00010 xm00010 xr00010 xt00010 xy00005 xq00005 xw00005 xc00005 xv00005 
        em00012 mr00012 rt00012 ty00006 yq00006 qw00006 wc00006 cv00006 emr mrt rty tyq
        yqw qwc wcv emrt mrty rtyq tyqw yqwc qwcv emrty mrtyq rtyqw tyqwc yqwcv emrtyq 
        mrtyqw rtyqwc tyqwcv emrtyqw mrtyqwc rtyqwcv emrtyqwc mrtyqwcv emrtyqwcv 
        exr00014 mxt00014 rxy00007 txq00007 yxw00007 qxc00007 wxv00007 exxt00016
        mxxy00008 rxxq00008 txxw00008 yxxc00008 qxxv00008 exxxy00009 mxxxq00009 
        rxxxw00009 txxxc00009 yxxxv00009 exxxxq mxxxxw rxxxxc txxxxv exxxxxw
        mxxxxxc rxxxxxv exxxxxxc mxxxxxxv exxxxxxxv'
    """  
    if d_pos_length != -1:
        d_pos_length = d_pos_length + 3
    maxwl = maxwl+1
    mindl = mindl+1
    maxdl = maxdl+2
    max_loop= max(maxwl, maxdl)
    lst1, length, lst2 = pos_prot_words(seq,max_loop=max_loop, pos_length=pos_length, pos_buffer=pos_buffer)
    #print lst1, length, lst2
    feature1 = " ".join(lst2[minwl:maxwl])
    #print feature1 + "\n"
    if maxdl>2:
        lst3 = pos_dist_relation(lst1,length,d_pos_length=d_pos_length, dis_buffer=dis_buffer, pos_buffer=pos_buffer)
        feature2 = " ".join(lst3[mindl:maxdl])
        #print feature2 + "\n"
        return (feature1 + " " + feature2)
    else:
        return feature1
    
def jackknife(bag, tag):
    count =0
    corr = 0
    x=bag
    y=tag
    accuracy = 0
    t=[]
    p=[]
    
    loo = LeaveOneOut()
    for a, b in loo.split(bag):
        
        x_train = x[a]; x_test = x[b]; y_train= y[a]; y_test =y [b]
        
        
        lr = LogisticRegression(C = 1, random_state=0)
    
    
        # fit the Logistic Regression model
        lr.fit(x_train, y_train)
    
        #predict the test set
        y_pred= lr.predict(x_test)
        
        #Number of misclassification
        c= (y_test != y_pred).sum()
        if c==0:
            corr= corr+1
            
        t.append(y_test)
        p.append(y_pred)
        #Show the accuracy score
        #print 'Accuracy: ' + str(accuracy_score(y_test, y_pred))
        accuracy +=(accuracy_score(y_test, y_pred))
        # mcc+=(matthews_corrcoef(y_test, y_pred) )
        count= count+1
    accuracy = accuracy_score(t,p)
    try:
        matthews_corr = matthews_corrcoef(t,p)
    except:
        matthews_corr = "not applicable for multiclasses"
    return accuracy, matthews_corr

