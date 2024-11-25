from modules.vikamigo import ViKaMiGo

def main():
    data_file = 'vikamigo_data/Intent_Data.csv'
    zero_shot_results_file = 'vikamigo_data/zero_shot_results.csv'
    non_zero_shot_results_file = 'vikamigo_data/non_zero_shot_results.csv'
    vikamigo = ViKaMiGo(data_file)

    # Initialize the ViKaMiGo model
    vikamigo = ViKaMiGo(data_file)

    # Train and log zero-shot model
    print("Training zero-shot model...")
    vikamigo.train_zero_shot(results_file=zero_shot_results_file)
    print(f"Zero-shot model results saved to {zero_shot_results_file}")

    # Train and log non-zero-shot model
    print("Training non-zero-shot model...")
    vikamigo.train_non_zero_shot(results_file=non_zero_shot_results_file)
    print(f"Non-zero-shot model results saved to {non_zero_shot_results_file}")

if __name__ == '__main__':
    main()


# from modules import DatasetHandler, IntentModel
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer #Text Vectorization TF-IDF (Term Frequency-Inverse Document Frequency)
# from sklearn.model_selection import train_test_split #Train-Test Split
# from sklearn.linear_model import LogisticRegression #classification model for text classification
# import pickle # save the model and the vectorizer

# """Better Text Preprocessing
# Text Cleaning: Ensure that your text is clean. Remove special characters, numbers, and irrelevant symbols that do not contribute to the classification task.
# Lemmatization or Stemming: Use lemmatization (e.g., using spaCy) to reduce words to their root form (e.g., "running" -> "run"). This can help reduce dimensionality and improve generalization.
# Stopword Removal: If you're not already, make sure to remove common stopwords like "and," "the," "is," etc., unless they carry semantic value.
# """
# #import spacy 
# """Data preprocessing: Loaded and cleaned the dataset.
# Text vectorization: Converted the text into numerical features using TF-IDF.
# Model training: Trained a Logistic Regression model.
# Saving the model: Saved both the model and the vectorizer for future use.
# """
# # Load dataset
# dataset_path = 'vikamigo_data/Intent_Data.csv'  # Ensure this is the correct path
# data = pd.read_csv(dataset_path)

# # View the first few rows to check the data
# print(data.head())
# print(f"Data shape before cleaning: {data.shape}")
# #Removing missing values 
# #Cleaning Data
# # Drop rows with missing text or intent
# data = data.dropna(subset=["text", "intent"])

# # nlp = spacy.load("en_core_web_sm")

# # def preprocess_text(text):
# #     doc = nlp(text)
# #     return " ".join([token.lemma_ for token in doc if not token.is_stop])

# # data["text"] = data["text"].apply(preprocess_text)

# # Check the shape of the cleaned data
# print(f"Data shape after cleaning: {data.shape}")

# # Initialize the vectorizer
# vectorizer = TfidfVectorizer(stop_words="english")

# # Fit the vectorizer on the dataset and transform the text data
# X = vectorizer.fit_transform(data["text"])

# # View the shape of the resulting feature matrix
# print(f"Feature matrix shape: {X.shape}")
# y = data["intent"]

# # Verify the labels
# print(y.head())

# # Split data into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Verify the shapes of the splits
# print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

# # Initialize the model
# model = LogisticRegression()

# # Train the model using the training data
# model.fit(X_train, y_train)

# # Check the model's accuracy on the test set
# accuracy = model.score(X_test, y_test)
# print(f"Model accuracy: {accuracy * 100:.2f}%")

# # Save the trained model and vectorizer
# with open('vikamigo_data/model.pkl', 'wb') as model_file:
#     pickle.dump(model, model_file)
    
# with open('vikamigo_data/vectorizer.pkl', 'wb') as vec_file:
#     pickle.dump(vectorizer, vec_file)


