import pandas as pd

class DatasetHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        
    def preprocess_data(self):
        # Preprocess text: remove special characters, lowercase text
        self.df['text'] = self.df['text'].str.replace('[^a-zA-Z\s]', '', regex=True)
        self.df['text'] = self.df['text'].str.lower()
        
    def get_data(self):
        return self.df[['text', 'intent']]



# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer

# class DatasetHandler:
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.vectorizer = TfidfVectorizer()
#         self.data = None

#     def load_data(self):
#         self.data = pd.read_csv(self.file_path)
#         print("Dataset loaded successfully.")

#     def preprocess(self):
#         X = self.data["query"]
#         y = self.data["intent"]
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         print("Data split into training and testing sets.")

#     def vectorize(self):
#         self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
#         self.X_test_vec = self.vectorizer.transform(self.X_test)
#         print("Data vectorized successfully.")
#         return self.X_train_vec, self.X_test_vec
