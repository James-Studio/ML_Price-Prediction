import numpy as py 
import pandas as pd 
from tensorflow import keras
from keras.datasets import mnist 


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train)

