##使用方法
*使用python语言实现对于支持向量机（SVM）特征选择的实现，特征选择算法为f-score,该程序的主要有点是可输入文件囊括了csv,libsvm,arff等在序列分类的机器学习领域常用到的格式，其中csv:最后一列为class,libsvm:第一列为class,arff:通常最后一列为类别，其中csv和libsvm中不存在开头，直接是使用的数据。*
	python train.py -i 1.csv,2.csv,3.libsvm,4.arff -c 5
	
- 其中train.py为程序名称
- -i :后面接文件名，可以为csv,libsvm,arff格式，多个文件也可以用，但建议不要，因为特征选择时间通常很长
- -c:后面5代表五折交叉验证
