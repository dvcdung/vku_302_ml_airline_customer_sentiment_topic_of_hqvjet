import torch
import numpy as np
import helper as JH
import constants as J
from tqdm import tqdm

from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer
from data_preprocessing import DataPreprocessor

class Embedder():
    def __init__(self):
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
        self.phobertModel: AutoModel = AutoModel.from_pretrained('vinai/phobert-base-v2').to(JH.device)
        self.encoded_data = None
        self.encoded_inputs = {}
        self.features = None

    # Tokenization ------------------------------------------< JOHN TAG
    def tokenize_plus(self, texts: list):
        print(f"[JV] ==========< TOKENIZATION >==========")
        len_texts = [len(text) for text in texts]
        print(f"[JV] Length => max: {np.max(len_texts)}, min: {np.min(len_texts)}, mean: {np.mean(len_texts)}, median: {np.median(len_texts)}")
        self.encoded_data = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256
        )

    def get_embedding(self):
        print(f"[JV] ==========< EMBEDDING >==========")
        self.phobertModel.eval()

        batch_size = 32
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(self.encoded_data['input_ids']), batch_size), desc="Calculating Embeddings"):
                batch_input_ids = self.encoded_data['input_ids'][i:i + batch_size].to(JH.device)
                batch_attention_mask = self.encoded_data['attention_mask'][i:i + batch_size].to(JH.device)
                
                outputs = self.phobertModel(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                batch_last_hidden_states = outputs.last_hidden_state
                embeddings.append(batch_last_hidden_states.cpu()) # Chuyển lại về CPU trước khi nối
        
        self.features = torch.cat(embeddings, dim=0)
        print(f"[JV] Shape of Last hidden states: {self.features.shape}")

# dp = DataPreprocessor()
# dp.readData(filename=J.RAW_DATA_FILE_NAME, drop=False, data_size=J.READ_DATA_SIZE)
# dp.preprocess_data()

# eb = Embedder()
# eb.tokenize_plus(dp.dataFrame['Content'].tolist())
# eb.get_embedding()
# print(eb.features.shape)