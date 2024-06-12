import helper as JH
import constants as J
import torch
from data_preprocessing import DataPreprocessor
from embedding import Embedder

class Predictor():
    def __init__(self):
        self.dp = DataPreprocessor()
        self.embedder = Embedder()
        self.svm_model = JH.load_model("svm_model.pkl")
        self.knn_model = JH.load_model("knn_model.pkl")
        self.nb_model = JH.load_model("nb_model.pkl")

    def predict_text(self, option, input):
        title_input = self.dp.normalize(input['title'], J.VN_lOWERCASE_CHARS)
        print(title_input)
        self.embedder.tokenize_plus([title_input])
        self.embedder.get_embedding()
        title = self.embedder.features
        
        content_input = self.dp.normalize(input['content'], J.VN_lOWERCASE_CHARS)
        print(content_input)
        self.embedder.tokenize_plus([content_input])
        self.embedder.get_embedding()
        content = self.embedder.features

        feature = torch.cat((title[:, :64], content), dim=1)
        # x_pred = feature[:, 0, :]
        # x_pred = torch.mode(feature, dim=1).values
        x_pred = torch.mean(feature, dim=1)
        # x_pred = feature.view(feature.size(0), -1)[:, :10000]

        print(f"[JV] ==========< PREDICTING >==========")
        if option == "svm":
            result = self.svm_model.predict(x_pred)
            print(result)
            return result
        elif option == "knn":
            result = self.knn_model.predict(x_pred)
            print(result)
            return result
        elif option == "nb":
            result = self.nb_model.predict(x_pred)
            print(result)
            return result

# predictor = Predictor()
# predictor.predict_text("svm", {'title': 'Đắt và mất vệ sinh', 'content': 'Giá thì đắt nhưng khâu vệ sinh lại không tốt. Không thể hài lòng với số tiền bỏ ra. Nên xem xét lại'})
