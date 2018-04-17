# -*- coding: utf-8 -*-
"""
Created on Sat Jul 01 02:01:08 2017

@author: Ashiqul
"""

from sys import argv
from classifier import classifier, predict
#file_lst= glob.glob("train/*.fasta")
#print optimization(file_lst)
script, train, test, n, k, np, kp, y, c = argv

n=int(n); k=int(k); np=int(np); kp =int(kp); y=int(y); c=int(c);

train = train[1:-1].split(",")

x, model, voc = classifier(train, n, k, np, kp, y, c, "5")

y_pred, ids, seqs = predict(test, model, voc, n, k, np, kp, y, c)

result = open("prediction.csv", "w")
for i, j, k in zip(y_pred, ids, seqs):
    result.write(i+", "+j+", "+k+"\n")
result.close()

