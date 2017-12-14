import pandas as pd
import numpy as ny
import sys
from sklearn.preprocessing import MinMaxScaler
import numpy as np

RFH_=pd.read_csv(sys.argv[1],header=None,index_col=None)#sys.argv[1])
end=len(RFH_.values[0])
RFH_=RFH_.values[:,0:end-1]
RFH_=pd.DataFrame(RFH_).astype(float)
#PseDNC_=pd.read_csv(sys.argv[2],header=None,index_col=None)
PseDNC_=pd.read_csv(sys.argv[2],header=None,index_col=None)
PseDNC_=pd.DataFrame(PseDNC_).astype(float)
print "first_file_num:",len(RFH_)
print "first_file_length:",len(RFH_.values[0])
print "second_file_num:",len(PseDNC_)
print "second_file_length:",len(PseDNC_.values[0])
RFH_PseDNC=pd.concat([RFH_,PseDNC_],axis=1)
print "output_file_num:",len(RFH_PseDNC)
print "output_file_len:",len(RFH_PseDNC.values[0])

scaler=MinMaxScaler()
scaler.fit_transform(np.array(RFH_PseDNC))
print "normalization"
# scaler.fit_transform(np.array(RFH_PseDNC))
pd.DataFrame(RFH_PseDNC).to_csv(sys.argv[3],header=None,index=False)
# print RFH_PseDNC