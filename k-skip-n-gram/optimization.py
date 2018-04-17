# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 21:13:54 2017

@author: Ashiqul
"""
from sys import argv
import glob
from optimizer import optimization
#file_lst= glob.glob("train/*.fasta")
#print optimization(file_lst)
script, file_lst = argv
trainset=file_lst[1:-1].split(",")
opt = open("optimization_result.txt", "w")
result = optimization(trainset)
result = result[1:-1].split(",")
opt.write("Value of \"n\"="+str(result[0])+";   ")
opt.write('\n')
opt.write("Value of \"k\"="+str(result[1])+";   ")
opt.write('\n')
opt.write("Value of \"np\"="+str(result[2])+";  ")
opt.write('\n')
opt.write("Value of \"kp\"="+str(result[3])+";   ")
opt.write('\n')
opt.write("Value of \"y\"="+str(result[4])+";   ")
opt.write('\n')
opt.write("Value of \"c\"="+str(result[5])+";   ")
opt.write('\n')
opt.close()
