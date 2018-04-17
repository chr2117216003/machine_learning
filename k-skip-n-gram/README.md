# README #


### What is this repository for? ###

* This repository contains the necessary file to use m-NGSG locally
* version 1.1
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* To set up m-NGSG in a local pc, download all the python files present the repository. In windows system, python is needed to be installed. 
* Dependencies: To run m-NGSG optimization, cross-validation(cv) or prediction, the following python packages are needed to be installed: 
-biopython
-numpy
-scikit-learn
-scipy



* How to run tests:
Important*** All the files should be under same folder and the following instrustion should be run in the command propmt.

 
1. Optimization:
To optimized the parameters for the specific training set run the optimization.py. Only one input parameter is needed to run optimization.py. An optimization_result.txt file will be created after the completion of computation. The input parameter should be list of fasta files different classes of protein sequence those will be used as the training set. Each compoment of the list will represent the training set of the a particular class. For example, if there x.fasta and y.fasta are two files those contains two different classes of proteins, then the one need to run "python optimization.py [class1.fasta,class2.fasta]" in the comand prompt. **There should not be any white space in the list of the training set**. Expect a couple of hours to a couple of days to get results based on number and size of the training set. 

2. Cross-validation and model construction 
Running cv.py will produce the cross-validation score for a centain training set and parameters. The inputs will be 
-the list different training classes in fasta format
-value of n [1-25]*;
-value of k [1-25]*;
-value of np [1-10]*;
-value of kp [1-10]*;
-value of y [1-10]*;
-value of c [1-10]*;
-type of the cross-validation [3,5, 10 or jackknife]
* the recommended ranges of the parameters.    
For example, if the training sets are class1.fasta, class2.fasta and class3.fasta, selected parameter values are n=10, k=10, np=2, kp=2, y=5, c=1, and cross-validation type is jackknife, then the comand prompt should be: 
"python cv.py [class1.fasta, class2.fasta,class3.fasta] 10 10 2 2 5 1 jackknife"
Important*: There should not be any whitespace in the training set list.
The result will be created in model_accuracy.txt in the same folder. 
3. Prediction:
To predinct the classes from a fasta file of unknown protein sequences, one needs to run the predict.py file. The input parameters for the file are:
-a list the fasta files of different classes as the training set
-a fasta file of the unknown samples 
-value of n [1-25]*;
-value of k [1-25]*;
-value of np [1-10]*;
-value of kp [1-10]*;
-value of y [1-10]*;
-value of c [1-10]*; 
*the recommended ranges of the parameters.  
For example, if the if training classes are class1.fasta, class2.fasta and class3.fasta, the unknown sample is unknown.fasta, and the parameter values are n=5, k=15, np=2, kp=4, y=6, c=2 then the input command promt will be: 
"python predict.py [class1.fasta,class2.fasta,class3.fasta] sample.fasta 5 15 2 4 6 2" 
The result will be created in prediction.csv in same folder. **The classes of prediction will be according to the position of a fasta file in the training set list. The first element will be considered as Class1, the second element will be considered as Class2 and so on.**  

### Who do I talk to? ###

* S M Ashiqul Islam (Mishu)
* email: s_islam@baylor.edu
* phone: +1 845 233 8126