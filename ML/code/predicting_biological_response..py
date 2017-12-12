import pandas as pd
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

Location = r'../data/biological_response..csv'
df=pd.read_csv(Location)
y=df['Activity'].values
X=df.loc[:,'D1':'D1776'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

def sigmoid(y_pred):
    return 1.0 / (1.0 + math.exp(-y_pred))

def log_loss_results(model, X, y):
    results = []
    for pred in model.staged_decision_function(X):
        results.append(log_loss(y, [sigmoid(y_pred) for y_pred in pred]))
    return results

def plot_loss(test_loss):
    min_loss_value = min(test_loss)
    min_loss_index = test_loss.index(min_loss_value)
    return min_loss_value, min_loss_index

def model_test(learning_rate):
    model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=250, verbose=False, random_state=241)
    model.fit(X_train, y_train)
    train_loss = log_loss_results(model, X_train, y_train)
    test_loss = log_loss_results(model, X_test, y_test)
    return plot_loss(learning_rate, test_loss, train_loss)

min_loss_results = {}
for learning_rate in [1, 0.5, 0.3, 0.2, 0.1]:
    min_loss_results[learning_rate] = model_test(learning_rate)

min_loss_value, min_loss_index = min_loss_results[0.2]
print('{:0.2f} {}'.format(min_loss_value, min_loss_index))

model = RandomForestClassifier(n_estimators=min_loss_index, random_state=241)
model.fit(X_train, y_train)
print(y_test)
y_pred = model.predict_proba(X_test)[:, 1]
print(model.predict_proba(X_test))
test_loss = log_loss(y_test, y_pred)
print(test_loss)
