import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression


def show_plot(y_pred, y_test):
    index = range(1,len(y_test)+1)
    plt.plot(index, y_pred, 'bo', label = "y_test")
    plt.plot(index, y_test, 'r-', label = "y_pred")
    plt.legend()
    plt.show()





data = pd.read_csv("dataset/advertising.csv")

x,y = data['wechat'], data['sales']
x,y= x.values, y.values
x,y = x.reshape(-1,1), y.reshape(-1,1)
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)

model = LinearRegression()
new_model = model.fit(x_train, y_train)
y_pred = new_model.predict(x_test)
show_plot(y_pred, y_test)
print("score of prediction : {:.2f} %".format(model.score(x_test,y_test)*100))