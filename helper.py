import re # regular
import timeit
import torch
import constants as J
import pickle
from sklearn.metrics import accuracy_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Remove duplicates letters - Example: '____' to '_'
def rmdup(character, input_string):
    return re.sub(rf'{re.escape(character)}+', character, input_string)

def normalize(str):
    return re.sub(r'\s+', ' ', str.replace('-', ' ')).lower()

def timeof(function, number):
    execution_time = timeit.timeit(lambda: function, number=number)
    print("TIME: ", execution_time)
    return execution_time

def count(list):
    item_counts = {}
    for item in list:
        if item in item_counts:
            item_counts[item] += 1
        else:
            item_counts[item] = 1
    item_counts = dict(sorted(item_counts.items(),  key=lambda item: item[1], reverse=True))

    return item_counts

def check(list, selected_list):
    return [1 if item_i in list else 0 for item_i in selected_list]

# FILE IO ------------------------------------------< JOHN TAG
def save(filename='test.txt', option='w', data=None):
    with open(J.DATASET_FOLDER_PATH + filename, 'w') as f:
        f.write(data)

def read(filename='test.txt', option='r', data=None):
    with open(J.DATASET_FOLDER_PATH + filename, option) as f:
        f.read(data)

def eval_model(y_test, y_pred): 
    print("Classificationn Report: \n", classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print("Model's accuracy: ", accuracy)
    return accuracy

def save_model(model, file_name):
    with open(J.RESOURCE_FOLDER_PATH + f"models/{file_name}", 'wb') as f:
        pickle.dump(model, f)
        print(f"[JV] Saved model into {file_name} file.")

def load_model(file_name):
    with open(J.RESOURCE_FOLDER_PATH + f"models/{file_name}", 'rb') as f:
        return pickle.load(f)