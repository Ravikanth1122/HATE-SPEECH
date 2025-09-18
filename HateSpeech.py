# hate_speech_pipeline_fixed.py
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords as nltk_stopwords

# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# Visualization / wordcloud
from wordcloud import WordCloud

# If running first time, uncomment to download nltk data:
# nltk.download('punkt')
# nltk.download('stopwords')

# ------- Load dataset -------
dataset = pd.read_csv("HateSpeechData_sample.csv")   # ensure this CSV is in the working dir
print("Dataset shape:", dataset.shape)
print(dataset.head())

# ------- Add text length column -------
# ensure column name 'tweet' exists; adjust if your csv uses different column name
dataset['text length'] = dataset['tweet'].astype(str).apply(len)

# Quick visualizations (optional)
sns.set(style="whitegrid")
g = sns.FacetGrid(data=dataset, col='class')
g.map(plt.hist, 'text length', bins=50)
plt.show()

sns.boxplot(x='class', y='text length', data=dataset)
plt.show()

dataset['class'].hist()
plt.xlabel('class')
plt.ylabel('count')
plt.show()

# ------- Preprocessing function -------
# Notes:
#  - Use .str.replace(..., regex=True) to avoid future warnings
#  - Avoid naming conflict with nltk.corpus.stopwords by using stop_words
stop_words = set(nltk_stopwords.words("english"))
stop_words_update = {"#ff", "ff", "rt"}
stop_words.update(stop_words_update)
stemmer = PorterStemmer()

def preprocess(series: pd.Series) -> pd.Series:
    """
    Take a pandas Series of raw tweets and return a Series of cleaned, tokenized & stemmed strings.
    """
    s = series.astype(str)

    # collapse multiple spaces
    s = s.str.replace(r'\s+', ' ', regex=True)

    # remove mentions
    s = s.str.replace(r'@[\w\-]+', '', regex=True)

    # remove URLs
    s = s.str.replace(r'http[s]?://\S+', '', regex=True)

    # remove non-letters (punctuation and digits), replace with space
    s = s.str.replace(r'[^a-zA-Z]', ' ', regex=True)

    # collapse spaces again and strip
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()

    # replace numbers pattern (if present) with string 'numbr' -- optional
    s = s.str.replace(r'\d+(\.\d+)?', 'numbr', regex=True)

    # lowercase
    s = s.str.lower()

    # tokenize (simple split), remove stopwords, apply stemming
    def tokenize_and_clean(text):
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words]
        tokens = [stemmer.stem(t) for t in tokens]
        return ' '.join(tokens)

    return s.apply(tokenize_and_clean)

# Apply preprocessing
dataset['processed_tweets'] = preprocess(dataset['tweet'])
print(dataset[['tweet', 'processed_tweets']].head(10))

# ------- WordClouds (optional) -------
all_words = ' '.join(dataset['processed_tweets'].values)
wc = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# For specific classes (assuming class labels 0/1 exist)
if 'class' in dataset.columns:
    for class_value, title in [(0, 'Hatred words (class=0)'), (1, 'Offensive words (class=1)')]:
        subset = dataset.loc[dataset['class'] == class_value, 'processed_tweets']
        if subset.empty:
            continue
        words = ' '.join(subset.values)
        wc = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(words)
        plt.figure(figsize=(10,7))
        plt.title(title)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.show()

# ------- Feature extraction (TF-IDF) -------
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=5, max_features=10000)
X = tfidf_vectorizer.fit_transform(dataset['processed_tweets'])
y = dataset['class'].astype(int)

# ------- Train/Test split (use a single split and reuse for fair comparison) -------
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Logistic Regression
lr = LogisticRegression(max_iter=2000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression")
print(classification_report(y_test, y_pred_lr))
acc_lr = accuracy_score(y_test, y_pred_lr)
print("Accuracy:", acc_lr)

# Random Forest (works with dense or sparse via sklearn)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest")
print(classification_report(y_test, y_pred_rf))
acc_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy:", acc_rf)

# Naive Bayes (GaussianNB requires dense arrays)
X_dense = X.toarray()   # convert entire matrix to dense (only do this if dataset fits in memory)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_dense, y, random_state=42, test_size=0.2)

nb = GaussianNB()
nb.fit(X_train_d, y_train_d)
y_pred_nb = nb.predict(X_test_d)
print("Naive Bayes (GaussianNB)")
print(classification_report(y_test_d, y_pred_nb))
acc_nb = accuracy_score(y_test_d, y_pred_nb)
print("Accuracy:", acc_nb)

# Linear SVC (works better on the sparse TF-IDF)
svc = LinearSVC(random_state=42, max_iter=5000)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
print("Linear SVC")
print(classification_report(y_test, y_pred_svc))
acc_svc = accuracy_score(y_test, y_pred_svc)
print("Accuracy:", acc_svc)

# ------- Bar plot comparison -------
models = ['Logistic', 'RandomForest', 'NaiveBayes', 'SVM']
accuracies = [acc_lr, acc_rf, acc_nb, acc_svc]
plt.figure(figsize=(8,5))
plt.bar(models, accuracies, alpha=0.6)
plt.ylabel('Accuracy')
plt.title('Algorithm Comparison')
plt.ylim(0,1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.show()
