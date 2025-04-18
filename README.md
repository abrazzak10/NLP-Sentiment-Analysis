# Sentiment-Analysis using NLP
This NLP project focuses on analyzing the sentiment of tweets. 
## Objective
The goal is to classify tweets into three categories: 
-Positive
-Negative
-Neutral

## I tried to get answer of the following Questions 
## Data Preprocessing
1. Clean the dataset using various preprocessing techniques.
## Tokenization:
2. Tokenize each tweet using appropriate nltk functions.
## Feature Extraction:
3. Apply TF-IDF Vectorization to transform tokens into numerical feature vectors.
## Model Training:
4. Train multiple ML models using twitter_training.csv:
-Support Vector Machine (SVM)
-Decision Tree
-Random Forest
-Multinomial Naive Bayes

## Hyperparameter Tuning:
5. Use grid search or cross-validation to find the best hyperparameters for each model.
## Model Evaluation:
6. Test the models using twitter_validation.csv.
7. Evaluate performance using metrics such as:
-Accuracy
-Precision
-Recall
-F1-score
-Confusion Matrix

## Visualization:

8. Visualize model performance using plots (e.g., confusion matrix heatmaps, bar charts).
9.Identify and recommend the best performing algorithm. 
## Tools & Libraries:
-Python
-Pandas, NumPy
-NLTK
-Scikit-learn
-Matplotlib, Seaborn
## Future work 
-Use deep learning models like LSTM or BERT
-Expand dataset for better generalization
