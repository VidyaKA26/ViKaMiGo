from modules.dataset_handler import DatasetHandler
from modules.intent_model import ZeroShotIntentModel, NonZeroShotIntentModel
from modules.api_server import ApiServer

class ViKaMiGo:
    def __init__(self, data_file):
        self.data_handler = DatasetHandler(data_file)
        self.zero_shot_model = ZeroShotIntentModel()
        self.non_zero_shot_model = NonZeroShotIntentModel()
        self.api_server = ApiServer()

    def train_zero_shot(self):
        # Example: Train the zero-shot model on the dataset (intent classification)
        data = self.data_handler.get_data()
        for _, row in data.iterrows():
            result = self.zero_shot_model.classify_intent(row['text'], ['Product Inquiry', 'Order Inquiry', 'Technical Support', 'Refund Inquiry'])
            print(f"Text: {row['text']} => Predicted Intent: {result['labels'][0]}")
    
    def train_non_zero_shot(self):
        # Format the dataset for non-zero-shot learning
        train_data = [(row['text'], {'cats': {row['intent']: 1}}) for _, row in self.data_handler.get_data().iterrows()]
        self.non_zero_shot_model.train_model(train_data)

    def deploy(self):
        self.api_server.start_server()



# from modules.dataset_handler import DatasetHandler
# from modules.intent_model import IntentModel
# from modules.api_server import APIServer

# class ViKaMiGo:
#     def __init__(self, dataset_path, model_path="model.pkl", vectorizer_path="vectorizer.pkl"):
#         self.dataset_path = dataset_path
#         self.model_path = model_path
#         self.vectorizer_path = vectorizer_path
#         self.dataset_handler = DatasetHandler(dataset_path)
#         self.intent_model = IntentModel()

#     def train_and_save(self):
#         self.dataset_handler.load_data()
#         self.dataset_handler.preprocess()
#         X_train_vec, X_test_vec = self.dataset_handler.vectorize()
#         self.intent_model.train(X_train_vec, self.dataset_handler.y_train)
#         self.intent_model.evaluate(X_test_vec, self.dataset_handler.y_test)
#         self.intent_model.save(self.model_path, self.vectorizer_path, self.dataset_handler.vectorizer)

#     def deploy_api(self):
#         model, vectorizer = IntentModel.load(self.model_path, self.vectorizer_path)
#         api_server = APIServer(model, vectorizer)
#         api_server.run()
