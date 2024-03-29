import pandas as pd
import numpy as np
import random
import pandas as pd
import seaborn as sn
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from joblib import load

def models():
  data= pd.read_csv("supervised_dataset.csv")

  for i in range(len(data)):
    if data['classification'].iloc[i] == 'normal':
        data.loc[i, 'Security Type'] = 'OAuth'
    elif data['classification'].iloc[i] == 'outlier':
        data.loc[i, 'Security Type'] = 'ApiKey'
  data['inter_api_access_duration(sec)']=data['inter_api_access_duration(sec)'].fillna(data['inter_api_access_duration(sec)'].mean())
  data['api_access_uniqueness']=data['api_access_uniqueness'].fillna(data['api_access_uniqueness'].mean())



  lab=LabelEncoder()
  data['type_ip']=lab.fit_transform(data['ip_type'])
  data['sources']=lab.fit_transform(data['source'])
  data['classifiction']=lab.fit_transform(data['classification'])
  
 
  x=data[['sequence_length(count)','vsession_duration(min)'
  ,'num_sessions','num_users' ,'num_unique_apis'
  ,'type_ip','sources']]
  y=data['classifiction']

  x_train,x_test,y_train,y_test=train_test_split(x,y)

  lr=LogisticRegression(max_iter=200)
  lr.fit(x_train,y_train)
  from joblib import dump
  dump(lr, 'lr.joblib')

  tree=DecisionTreeClassifier(criterion='entropy',max_depth=1)
  tree.fit(x_train,y_train)
  from joblib import dump
  dump(tree, 'tree.joblib')
  clf_lda = load('tree.joblib')
  print("LDA score and classification:")
  print(clf_lda.score(x_test, y_test))
  print(clf_lda.predict(x_test))
  clf_nn = load('lr.joblib')
  print("NN score and classification:")
  print(clf_nn.score(x_test, y_test))
  print(clf_nn.predict(x_test))

if __name__ == '__main__':
    models()
