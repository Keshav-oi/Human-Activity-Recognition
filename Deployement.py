import os
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

classes = {"Walking": 1, "WalkingUpstairs": 2, "WalkingDownstairs": 3, "Sitting": 4, "Standing": 5, "Laying": 6}

X_test_collected = []
y_test_collected = []
print("Loading Test data....")

for folder in classes.keys():
    files = os.listdir("D:\\Third Year\\Sem II\\ML\\A1\\Mini Project\\Collected-Data\\" + folder)
    for file in files:
        path = "D:\\Third Year\\Sem II\\ML\\A1\\Mini Project\\Collected-Data\\" + folder + "\\" + file
        try:
            df = pd.read_excel(path)
        except:
            df = pd.read_csv(path)
        df = df[:500]
        df.columns = ['time', 'ax', 'ay', 'az', 'at']
        X_test_collected.append(np.array(df['at'].values))
        y_test_collected.append(classes[folder])

X_test_collected = np.array(X_test_collected)
print(X_test_collected.shape)
x = []
y = []
for i in range(X_test_collected.shape[0]):
    for j in range(X_test_collected.shape[1]):
        x_data = X_test_collected[i, j]
        x.append(x_data)
        y.append(y_test_collected[i])
x_test = np.array(x).reshape(-1, 1)
y_test = np.array(y).reshape(-1, 1)
print("Test Data Loaded successfully!!!")

X_train_collected = []
y_train_collected = []
print("\nLoading Train data....")

for folder in classes.keys():
    files = os.listdir("D:\\Third Year\\Sem II\\ML\\A1\\Mini Project\\Combined\\Train\\" + folder)
    for file in files:
        path = "D:\\Third Year\\Sem II\\ML\\A1\\Mini Project\\Combined\\Train\\" + folder + "\\" + file
        try:
            df = pd.read_excel(path)
        except:
            df = pd.read_csv(path)
        df = df[:500]
        df.columns = ['ax', 'ay', 'az']
        at = ((df['ax'] ** 2 + df['ay'] ** 2 + df['az'] ** 2) ** 0.5)
        X_train_collected.append(at)
        y_train_collected.append(classes[folder])

X_train_collected = np.array(X_train_collected)
print(X_train_collected.shape)
x_train = []
y_train = []

for i in range(X_train_collected.shape[0]):
    for j in range(X_train_collected.shape[1]):
        x_data = X_train_collected[i, j]
        x_train.append(x_data)
        y_train.append(y_train_collected[i])
x_train = np.array(x_train).reshape(-1, 1)
y_train = np.array(y_train).reshape(-1, 1)
print("Train data loaded successfully")

print("\nTraining Decision Tree.....")
Recognizer = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=20, criterion='gini', random_state=42)
Recognizer.fit(x_train, y_train)
print("Predicting from tree....")
y_pred = Recognizer.predict(x_test)
print("Predictions completed!!!!")
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the Decision Tree for max_depth = {10} is:", accuracy)
con_mat = confusion_matrix(y_test, y_pred, labels=Recognizer.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=Recognizer.classes_)
disp.plot()
plt.show()
