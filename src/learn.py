import pickle
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

DELTA = 100

x = pd.read_csv('/tmp/hh',sep=' ')
res = (x['evaluation'] + DELTA <= x['alpha'])

data = x.drop(['evaluation','fen','castle','enp','ply1','ply2','who','step13', 'note'],
              axis=1)

x_train,x_test,y_train,y_test = model_selection.train_test_split(data,res.values,test_size=0.1)

# cfitter = HistGradientBoostingClassifier(verbose=1,max_iter=50,max_leaf_nodes=15)
# cmodel = cfitter.fit(x_train,y_train)

# print('score',cmodel.score(x_test,y_test))
# print('raw',sum(y_train) / y_train.shape[0])

def get_model(i,n,verbose=0):
  cf = HistGradientBoostingClassifier(max_iter=i,max_leaf_nodes=n,verbose=verbose)
  x_t,x_v,y_t,y_v = model_selection.train_test_split(data,res.values,test_size=0.1)
  return cf.fit(x_t,y_t),x_v,y_v
  
def score(i,n,verbose=0):
  m,x,y = get_model(i,n,verbose=verbose)
  return m.score(x,y)

def show(model,x,y,color='b'):
  cscores = cross_val_predict(model,x,y,cv=3,method='predict_proba')[:,1]
  fpr_c,tpr_c,thresh_c = roc_curve(y,cscores)
  plt.plot(fpr_c,tpr_c,color+'-')
  return None

def score_and_show(i,n,verbose=0,color='b'):
  m,x,y = get_model(i,n,verbose=verbose)
  show(m,x,y,color=color)
  return None

# various ways of dumping things out, a field at a time

def dump_pickle(model,file):
  pickle.dump(model,open(file,'wb'))
  return None

def dump_vec(vec,file):
  print(len(vec),file=file)
  for x in vec:
    if type(x) is np.ndarray: print(x[0],file=file)
    else: print(x,file=file)
  return None

# dump a model (regression or classification) to an open file

def dump_model(model,file):
  if type(model._baseline_prediction) is np.ndarray:
    dump_vec(model._baseline_prediction,file)
  else: dump_vec([model._baseline_prediction],file)
  print(len(model._predictors),file=file)
  for p in model._predictors:
    for pred in p:
      nodes = pred.nodes
      print(nodes.shape[0],file=file)
      for n in nodes: print(n[0],n[2],n[3],n[4],n[5],n[6],n[9],file=file)
  return None

# dump a model given a file name

def dumpc(model,file='dump'):
  f = open(file,'w')
  dump_model(model,f)
  return None

def dump(model,file):
  dumpc(model,file + '.txt')
  dump_pickle(model,file + '.pkl')

# m = pickle.load(open('step13.pkl','rb'))
# d = pd.read_csv('step13.input',sep=' ')
