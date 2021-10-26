import pandas as pd
import numpy as np
import pickle
df=pd.read_csv('Iris.csv')
df['Species'].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[0,1,2],inplace=True)
x = np.array(df.iloc[:, 1:5])
y = np.array(df.iloc[:, -1])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
da=model.predict([[5,2.3,3.3,1]])
pickle.dump(model, open('iri.pkl', 'wb'))