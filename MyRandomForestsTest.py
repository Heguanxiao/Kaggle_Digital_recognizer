import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('train.csv')
X_tr = data.values[:,1:].astype(float)
y_tr = data.values[:,0]

scores = list()
scores_std = list()

n_trees = [125]
for n_tree in n_trees:
	clf = RandomForestClassifier(n_estimators = n_tree)




clf.fit(X_tr,y_tr)
test = pd.read_csv('test.csv')
X_te = test.values[:,0:]
y_te = clf.predict(X_te)
writer = open('predict.csv','w')
count = 1
writer.write('"ImageId","Label"\n')
for p in y_te:
	writer.write(str(count)+',"'+str(p)+'"\n')
	count = count +1