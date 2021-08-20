import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 顯示學習曲線
def show_map(y_pred, y_test): # 显示训练过程中的学习曲线
    index = range(1, len(y_pred) + 1)
    plt.figure(figsize=(12,4))
    plt.plot(index, y_test, 'bo', label='y_test')
    plt.plot(index, y_pred, 'r', label='y_predict')
    plt.title('Prediction Result')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    

def show_correlation(x_train, y_train) :
    plt.plot(x_train,y_train,'r.', label='Training data') 
    plt.xlabel('Wechat Ads')
    plt.ylabel('Sales')
    plt.legend() 
    plt.show() 



data = pd.read_csv('dataset/advertising.csv')
print(data)

print(data.corr())

x = np.array(data['wechat'])
y = np.array(data['sales'])

print("x.shape:",x.shape)
print("y.shape:",y.shape)

# reshape : 2D tensor is essential in regression problem !!! 
x = x.reshape(-1,1)
y = y.reshape(-1,1)

# divide the dataset into training set and testing set !!!
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state = 0)

# 資料歸一化
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

# Deep Learning 
# Build Dense Layer !!!

# 架設網路模型
ann = Sequential() # 創建一個序列ann模型
ann.add(Dense(units = 3, input_dim = 1,activation = 'relu')) # 增加輸入層
ann.add(Dense(units = 1, activation = 'sigmoid')) # 增加輸出層
ann.summary() # 顯示網路模型

# 編譯神經網路
ann.compile(optimizer = 'adam',loss = 'mean_squared_error',metrics=['acc'])

# 訓練網路
history = ann.fit(x_train, y_train, epochs = 5, batch_size = 20, validation_data = (x_test, y_test))

# 預測數值
y_pred = ann.predict(x_test)

# 印圖
y_pred = y_pred.reshape(1,-1)
y_test = y_test.reshape(1,-1)
x_test = x_test.reshape(1,-1)
y = np.r_[y_pred,y_test]
x_y = np.r_[x_test,y_test]
cor_1 = np.corrcoef(y)[0, 1]
cor_2 = np.corrcoef(x_y)[0, 1]
print("cor_1: ",cor_1)
print("cor_2: ",cor_2)

y_pred = y_pred.reshape(-1,1)
y_test = y_test.reshape(-1,1)
x_test = x_test.reshape(-1,1)
show_map(y_pred, y_test)
show_correlation(x_test, y_test)



# 畫圖
"""
plt.plot(history, 'g--',label = 'loss curve')
plt.legend()
plt.show()
"""
