import pickle
import torch 
import numpy as np
import constants as J
from data_preprocessing import DataPreprocessor
from embedding import Embedder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import helper as JH

class LogisticModel():
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def build_model(self):
        print(self.x_data.shape, self.y_data.shape)
        # y_data = np.array(dp.dataFrame['Genres'].tolist())
        x_train, x_test, y_train, y_test = train_test_split(self.x_data, self.y_data, test_size=0.2)       
        model = MultiOutputClassifier(LogisticRegression())
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return model, JH.eval_model(y_test, y_pred)

    def train_model(self, train_loop):
        max = 0.0
        model = None
        for i in range(0, train_loop):
            # BUILD MODEL
            cur_model, accuracy = self.build_model()
            if (accuracy >= max):
                model = cur_model
                max = accuracy
        
        with open(J.RESOURCE_FOLDER_PATH + 'models/logistic_regression_model.pkl', "wb") as f:
            pickle.dump(model, f)

        print(max)

    def predict(x_test):
        with open(J.RESOURCE_FOLDER_PATH + 'models/logistic_regression_model.pkl', 'rb') as file:
            model = pickle.load(file)

        return model.predict(x_test)