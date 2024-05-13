import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
from constant import *

class DTree():
    def __init__(self):
        self.x = None
        self.y = None
        self.getDataset()

    def getDataset(self):
        features = np.load(FEATURES_FILE_PATH)
        self.x = features[:, 0, :]
        self.y = np.array(pd.read_csv(DATASET_FILE_PATH)['Rating'][:8000])

    def train(self, max_iter):
        last_model = None
        max_score = 0

        for i in range(0, max_iter):
            X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            # Print report ------------------------------------------< JOHN TAG
            y_pred = model.predict(X_test)
            print(y_pred)
            print(classification_report(y_test, y_pred))
            accur_score = accuracy_score(y_test, y_pred)
            if(accur_score > max_score):
                last_model = model
                max_score = accur_score

        # Export model ------------------------------------------< JOHN TAG
        with open(TRAINED_MODEL_FILE_PATH, 'wb') as file:
            pickle.dump(last_model, file)
        
        # Print best result ------------------------------------------< JOHN TAG
        print("> MAX_ACCURACY_SCORE: ", max_score)
        print("> MODEL: \n", last_model)

    def predict(self, x_pre):
        # Đọc mô hình từ file PKL
        with open(TRAINED_MODEL_FILE_PATH, 'rb') as file:
            loaded_model = pickle.load(file)
        y_pre = loaded_model.predict(x_pre)
        return y_pre[0]

d_tree = DTree()
d_tree.train(20)
# print(d_tree.predict(d_tree.x[:1, :]))
        