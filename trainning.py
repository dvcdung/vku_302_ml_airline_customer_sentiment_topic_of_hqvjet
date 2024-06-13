import torch
import constants as J
import helper as JH
from models.support_vector_machine_model import SVMClassifier
from models.knn_model import KNNClassifier
from models.naive_bayes_model import NaiveBayesClassifier

class Trainer():
    def __init__(self):
        dataset = torch.load(J.DATASET_FOLDER_PATH + J.DATASET_FILE_NAME)
        self.title = dataset['title']
        self.content = dataset['content']
        self.x_data = torch.cat((self.title[:, :64], self.content), dim=1)
        self.y_data = dataset['label']

    def train(self, option, epochs=1):
        print(f"[JV] ==========< TRAINING >==========")
        if option == "svm":
            # x_data = torch.mode(self.x_data, dim=1).values
            # x_data = self.x_data[:, 0, :]
            x_data = torch.mean(self.x_data, dim=1)
            # x_data = self.x_data.view(self.x_data.size(0), -1)[:, :10000]
            y_data = self.y_data
            classificer = SVMClassifier(kernel='linear', C=5, random_state=42) # C is soft-margin parameter -> custom
            classificer.build()
            classificer.train_model(x_data, y_data, test_size=0.2, epochs=epochs)
        elif option == "knn":
            # x_data = torch.mode(self.x_data, dim=1).values
            # x_data = self.x_data[:, 0, :]
            x_data = torch.mean(self.x_data, dim=1)
            # x_data = self.x_data.view(self.x_data.size(0), -1)[:, :10000]
            y_data = self.y_data
            classificer = KNNClassifier(max_n_neighbors=100, weights='uniform', algorithm='auto', random_state=42)
            classificer.train_model(x_data, y_data, test_size=0.2, epochs=epochs)
        elif option == "nb":
            # x_data = torch.mode(self.x_data, dim=1).values
            # x_data = self.x_data[:, 0, :]
            x_data = torch.mean(self.x_data, dim=1)
            # x_data = self.x_data.view(self.x_data.size(0), -1)[:, :100000]
            y_data = self.y_data
            classificer = NaiveBayesClassifier(random_state=42)
            classificer.build()
            classificer.train_model(x_data, y_data, test_size=0.2, epochs=epochs)
            
trainer = Trainer()
# trainer.train("logistic_model", epochs=50)
trainer.train("nb", epochs=1)

# print(JH.read(filename=J.DATASET_FILE_NAME))