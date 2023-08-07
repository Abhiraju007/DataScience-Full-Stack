import numpy as np
import pandas as pd
from KNearestNeighbor import KNearestNeighbors

data = pd.read_csv('Social_Network_Ads.csv')

#print(data.head())

X = data.iloc[:,2:4].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#print(X_train.shape)

#An object of KNN

knn = KNearestNeighbors(k=3)
knn.fit(X_train,y_train)

def predict_new():
    age = int(input("Enter a Age: "))
    salary = int(input("Enter a salary: "))
    X_new = np.array([[age],[salary]]).reshape(1,2)
    X_new = scaler.transform(X_new)

    result=knn.predict(X_new)

    if result==0:
        print("Will not purchase")
    else:
        print("Will Purchase")


predict_new()

