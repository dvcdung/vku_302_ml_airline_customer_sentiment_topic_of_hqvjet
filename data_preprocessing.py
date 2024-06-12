import pandas as pd
import numpy as np
import re
import helper as JH
import constants as J
from pyvi import ViTokenizer

class DataPreprocessor():
    def __init__(self):
        self.dataFrame = None

    # READ FROM DATASET FILE ------------------------------------------< JOHN TAG
    def readData(self, filename="", drop=False, columns_selected=None, data_size=None):
        print(f"[JV] Reading data from {filename}")
        df = pd.read_csv(J.RAW_DATA_FOLDER_PATH + filename)
        # _note_ > Drop duplicates
        df.drop_duplicates(subset=['Content'], keep=False, inplace=True)
        # _note_ > Get the columns needed to solve the problem
        if (columns_selected): df = df[columns_selected]
        # _note_ > Drop rows containing NaN values
        self.dataFrame = df.dropna()[:data_size]
        print(f"[JV] Shape: {self.dataFrame.shape}")

    # Get preprocessed data for next step ------------------------------------------< JOHN TAG
    def getData(self, limit=None):
        return self.dataFrame[:limit] if limit else self.dataFrame
    
    # Normalize text ------------------------------------------< JOHN TAG
    def normalize(self, text, regex_chars='', regex_spec_chars=''):
        # regex_chars -> represent letters that need to be kept in the sentences
        # regex_spec_chars -> represents special characters that need to be kept in the sentences
        if (len(regex_spec_chars) > 0):
            # _note_ > loop to handle special characters
            text = re.sub(rf'[^0-9{regex_chars}\{regex_spec_chars}]', ' ', text.lower())
            for ch in regex_spec_chars:
                list = text.split(ch)
                list = [' '.join(item.split()) for item in list]
                list = [item for item in list if (item != '' and item != ' ')]
                text = f'{ch} '.join(list)
        else:
            # _note_ > loop to handle special characters
            text = re.sub(rf'[^0-9{regex_chars}]', ' ', text.lower())

        text = text.replace('-', ' -')
        text = ViTokenizer.tokenize(text)
        stopwords = ['cô', 'tôi', 'mình', 'ní', 'ngộ', 'cá_nhân', 'trai', 'gái', 'ông', 'bác', 'chú', 'dì', 'gì', 'hay_là', 'đôi_khi', 'thật', 'nhiều', 'hãy', 'này', 'nay', 'đó', 'đây', 'hoàn_toàn', 'mà', 'hình_như', 'thật_sự', 'thực_sự', 'thực_ra', 'từ_đó', 'từ_đây', 'hay_là', 'bỗng', 'bỗng_dưng', 'nhưng', 'lại', 'nhé']
        text_without_stopwords = [word for word in text.split() if word not in stopwords]

        return ' '.join(text_without_stopwords)
    
    # Preprocess data ------------------------------------------< JOHN TAG
    def preprocess_data(self):
        print(f"[JV] ==========< DATA PREPROCESSING >==========")
        # self.dataFrame =  self.dataFrame.apply(lambda column: column.apply(lambda row: self.normalize(str(row), J.VN_lOWERCASE_CHARS, '-,.')))
        self.dataFrame['Rating'] = self.dataFrame['Rating'].map({1: 0, 2: 1, 3: 2})
        self.dataFrame['Title'] = self.dataFrame['Title'].apply(lambda row: self.normalize(str(row), J.VN_lOWERCASE_CHARS))
        self.dataFrame['Content'] = self.dataFrame['Content'].apply(lambda row: self.normalize(str(row), J.VN_lOWERCASE_CHARS))
        
        print(f"[JV] Data for embedding:\n", self.dataFrame)
        print(f"[JV] Shape: {self.dataFrame.shape}")

# dp = DataPreprocessor()
# dp.readData(filename=J.RAW_DATA_FILE_NAME, drop=False, data_size=J.READ_DATA_SIZE)
# dp.preprocess_data()
