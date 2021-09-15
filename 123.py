import cv2
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import pandas as pd
import matplotlib.pyplot as plt
import os,time,ssl

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

X,y=fetch_openml("mnist_784",version=1,return_X_y=True)
# print(X[0])
print(pd.Series(y).value_counts())

classes=["0","1","2","3","4","5","6","7","8","9"]
nclasses=len(classes)
# print(nclasses)

# 7500 train 2500 test
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=2500,train_size=7500)

x_train_scaled=x_train/255
x_test_scaled=x_test/255

model=LogisticRegression(solver="saga",multi_class="multinomial")
model.fit(x_train_scaled,y_train)
# accuracy
y_predict=model.predict(x_test_scaled)
accuracy= accuracy_score(y_test,y_predict)
print(accuracy)
