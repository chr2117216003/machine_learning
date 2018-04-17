# -*- coding: utf-8 -*-
"""
Created on Sat Jul 01 01:01:30 2017

@author: Ashiqul
"""

from sys import argv
from classifier import classifier

script, file_list, n, k, np, kp, y, c, cv = argv

file_list = file_list[1:-1].split(",")


n=int(n)
k=int(k)
np=int(np)
kp=int(kp)
y=int(y)
c=int(c)

result = open("model_accuracy.txt", "w")
accuracy, x, y = classifier(file_list, n, k, np, kp, y, c, cv)
result.write("The "+cv+" fold cross-validation of the model is: "+str(accuracy*100)+"%")
result.close()
