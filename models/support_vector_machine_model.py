import torch
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import constants as J
import helper as JH

class SVMClassifier:
    def __init__(self, kernel='linear', C=1.0, random_state=42):
        self.kernel = kernel
        self.C = C
        self.random_state = random_state
        self.model = None
        self.scaler = None

    def build(self):
        self.model = svm.SVC(kernel=self.kernel, C=self.C, random_state=self.random_state)
        self.scaler = StandardScaler()

    def train_model(self, features, labels, test_size=0.2, epochs=1):
        for i in range(0, epochs):
            # Convert features and labels to numpy arrays if they are PyTorch tensors
            if isinstance(features, torch.Tensor):
                features = features.numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=self.random_state)

            # Standardize the features
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            # Train the SVM model
            self.model.fit(X_train, y_train)

            # Predict on the test set and print the classification report
            y_pred = self.model.predict(X_test)
            print(classification_report(y_test, y_pred, target_names=['Tiêu cực', 'Trung lập', 'Tích cực']))

            # Export model
            JH.save_model(self.model, 'svm_model.pkl')

    def predict(self, new_data):
        # Convert new data to numpy array if it is a PyTorch tensor
        if isinstance(new_data, torch.Tensor):
            new_data = new_data.numpy()

        # Standardize the new data
        new_data = self.scaler.transform(new_data)

        # Predict the labels for the new data
        predictions = self.model.predict(new_data)
        return predictions
        