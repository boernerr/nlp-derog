import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# import tabulate Don't have this
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score

from copied_project.src import text_normalizer

def identity(words):
    return words

def create_pipeline(estimator, reduction=False):
    steps = [
    ('normalize', text_normalizer.TextNormalizer()),
    ('vectorize', TfidfVectorizer(
    tokenizer=identity, preprocessor=None, lowercase=False
    ))
    ]
    if reduction:
        steps.append(('reduction', TruncatedSVD(n_components=10000)
        ))
    # Add the estimator
    steps.append(('classifier', estimator))
    return Pipeline(steps)

models = []
for form in (LogisticRegression, MultinomialNB, SGDClassifier):
    models.append(create_pipeline(form(), True))
    models.append(create_pipeline(form(), False))

# this accepts train/test splits. Need to work this in with KFolds splitting
for model in models:
    model.fit(train_docs, train_labels)

def model_accuracy(loader):
    for model in models:
         scores = [] # Store a list of scores for each split
         for X_train, X_test, y_train, y_test in loader: # loader needs to be a corpus loader instance
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            scores.append(score)
        print("Accuracy of {} is {:0.3f}".format(model, np.mean(scores)))

model = create_pipeline(SGDClassifier(), False)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, labels=labels))

fields = ['model', 'precision', 'recall', 'accuracy', 'f1']
table = []
for model in models:
    scores = defaultdict(list) # storage for all our model metrics
    # k-fold cross-validation
    for X_train, X_test, y_train, y_test in loader:
         model.fit(X_train, y_train)
         y_pred = model.predict(X_test)
         scores['precision'].append(precision_score(y_test, y_pred))
         scores['recall'].append(recall_score(y_test, y_pred))
         scores['accuracy'].append(accuracy_score(y_test, y_pred))
         scores['f1'].append(f1_score(y_test, y_pred))

# Aggregate our scores and add to the table.
    row = [str(model)]
    for field in fields[1:]
        row.append(np.mean(scores[field]))
    table.append(row)

# Sort the models by F1 score descending
table.sort(key=lambda row: row[-1], reverse=True)
print(tabulate.tabulate(table, headers=fields))