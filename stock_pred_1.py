import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from pandas.core.tools.datetimes import Scalar
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM 

def show_stock_price(data):
    x_axis = []
    for i in range(1,len(data["date"])+1):
        x_axis.append(i)
    new_date = list(data["date"])
    plt.plot(data["date"], (data["high"]+ data["low"])/2, 'b-', label = "price of daily average")
    # need to turn the date(pandas dataframe) into a list 
    # Attention : the amount of first and second parameter has to be the same !!!
    plt.xticks(x_axis[::100], new_date[::100], rotation=45)
    plt.xlabel("date")
    plt.ylabel("stock price")
    plt.legend() # print the additional bar in the chart 
    plt.show()

def compare_graph(data,avg_price, y_pred):
    x_axis = []
    len_pred = len(y_pred)
    len_data = data.shape[0]
    for i in range(1,len(data["date"])+1):
        x_axis.append(i)
    new_date = list(data["date"])
    plt.plot(data["date"], avg_price,'b-', label = "actual price")
    plt.plot(data["date"][len_data-len_pred:len_data], y_pred,"r-",label = "predict price")
    plt.xticks(x_axis[::100], new_date[::100], rotation=45)
    plt.xlabel("date")
    plt.ylabel("stock price")
    plt.legend()
    plt.show()

# create x_data and y_data 
def create_dataset(avg_data, data_step):
    x_data = []
    y_data = []
    for i in range(0,len(avg_data)-data_step):
        # 不是一直讓資料進行覆蓋，而是要往後增加
        x = avg_data[i:i+data_step,0] # index : i ~ (i+data_step-1)
        y = avg_data[i+data_step,0] # index : i+data_step 
        x_data.append(x)
        y_data.append(y)
    x_data = np.array(x_data)
    y_data = np.array(y_data).reshape(-1,1)
    return x_data, y_data

    
# read the data
data = pd.read_csv("dataset/AAP_data.csv")
#show_stock_price(data)
print("data sample:\n",data.head())

# use average stock price to predict
avg_price = np.array((data["open"] + data["close"])/2)
avg_price = avg_price.reshape(-1,1)
print(avg_price.shape)


# split train and test data
# train_size = 0.6, test_size = 0.4
train_num = round(avg_price.shape[0]*0.6) # 四捨五入
test_num = avg_price.shape[0] - train_num
print("train_num : ", train_num)
print("test_num : ", test_num)
# index : 0~1258
train_data = avg_price[0:train_num, 0] #index : 0~754
test_data = avg_price[train_num:train_num + test_num + 1, 0] #index : 755~1259
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)
print("train_data.shape : ", train_data.shape)
print("test_data.shape : ", test_data.shape)

# create x_train, y_train, x_test, y_test dataset
# data_step = 100 
x_train, y_train = create_dataset(train_data, 100)
x_test, y_test = create_dataset(test_data, 100)
print("train_num : ", x_train.shape, y_train.shape)
print("test_num : ", x_test.shape, y_test.shape)

# data preprocessing 
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

# transform to 3D tensor 
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

# LSTM model !!!
# model build 
model = Sequential()
model.add(LSTM(200, return_sequences = True, input_shape = (100,1)))
model.add(LSTM(400, return_sequences = True, input_shape = (100,1)))
model.add(LSTM(600))
model.add(Dense(1)) 
# model compile 
model.compile(loss = "mean_squared_error", optimizer = "adam")
model.summary()
# model train
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 1, batch_size = 100, verbose = 1)
# model predict 
y_pred = model.predict(x_test, verbose = 1, use_multiprocessing=True)
print("result :\n", y_pred)

# comparsion
compare_graph(avg_price, y_pred)