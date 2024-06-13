import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import constants as J
import helper as JH

class KNNClassifier:
    def __init__(self, max_n_neighbors=5, weights='uniform', algorithm='auto', random_state=42):
        self.max_n_neighbors = max_n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.random_state = random_state
        self.model = None
        self.scaler = None

    def train_model(self, features, labels, test_size=0.2, epochs=1):
        self.scaler = StandardScaler()
        max_accuracy = 0
        n_neighbors_selected = 0
        for n_neighbors in range(1, self.max_n_neighbors):
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=self.weights, algorithm=self.algorithm)

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

            # Train the KNN model
            model.fit(X_train, y_train)

            # Predict on the test set and print the classification report
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(classification_report(y_test, y_pred, target_names=['Tiêu cực', 'Trung lập', 'Tích cực']))
            print(f"Accuracy: {accuracy}")
            if accuracy > max_accuracy:
                self.model = model
                max_accuracy = accuracy
                n_neighbors_selected = n_neighbors

        # Export model
        JH.save_model(self.model, 'knn_model.pkl')
        print(f"Max accuracy: {max_accuracy} when n_neighbors = {n_neighbors_selected}" )

    def predict(self, new_data):
        # Convert new data to numpy array if it is a PyTorch tensor
        if isinstance(new_data, torch.Tensor):
            new_data = new_data.numpy()

        # Standardize the new data
        new_data = self.scaler.transform(new_data)

        # Predict the labels for the new data
        predictions = self.model.predict(new_data)
        return predictions