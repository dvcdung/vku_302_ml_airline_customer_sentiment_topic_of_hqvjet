from data_preprocessing import DataPreprocessor
from embedding import Embedder
import constants as J
import numpy as np
import torch

dp = DataPreprocessor()
dp.readData(filename=J.RAW_DATA_FILE_NAME, drop=False, data_size=J.READ_DATA_SIZE)
dp.preprocess_data()

embedder1 = Embedder()
embedder1.tokenize_plus(dp.getData()['Title'].tolist())
embedder1.get_embedding()

embedder2 = Embedder()
embedder2.tokenize_plus(dp.getData()['Content'].tolist())
embedder2.get_embedding()

def export_dataset():
    print(f"[JV] ==========< DATASET EXPORTING >==========")
    dataset = {'title': embedder1.features,
               'content': embedder2.features,
               'label': dp.dataFrame['Rating']}
    print(dataset['label'])

    torch.save(dataset, J.DATASET_FOLDER_PATH + J.DATASET_FILE_NAME)
    print(f"[JV] Exported dataset to {J.DATASET_FILE_NAME} file")

export_dataset()