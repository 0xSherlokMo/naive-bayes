from helpers import isCategorized
import numpy as np 
import pandas as pd 
import pickle
import category_encoders
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


df = pd.read_csv("./dataset.csv")

df.columns = [
    'age', 'workclass', 'fnlwgt',
    'education', 'education_num', 'marital_status', 
    'occupation', 'relationship', 'race', 
    'sex', 'capital_gain', 'capital_loss', 
    'hours_per_week', 'native_country', 'income'
]
print(df.head)

## Preprocessing data

print("Started PreProcessing")

print("Feature Selection")
df.drop(['fnlwgt'], axis=1, inplace=True)
df.drop(['race'], axis=1, inplace=True)

categorizedColumns = []
for column in df.columns:
    if isCategorized(df[column].dtype) == False:
        continue
    categorizedColumns.append(column)

print("checking missing values") 
print(df.isnull().sum())



for column in categorizedColumns:
    df[column].replace(' ?', np.nan, inplace=True)

#filling missing data
for column in categorizedColumns:
    df[column].fillna(df[column].mode()[0], inplace=True)


X = df.drop(['income'], axis=1)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=4)
print(X_train.shape, X_test.shape)

print("One hot encoding")
categorizedColumns = []
for column in X_train.columns:
    if isCategorized(X_train[column].dtype) == False:
        continue
    categorizedColumns.append(column)

encoder = category_encoders.OneHotEncoder(cols=categorizedColumns)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)


print("Standardizing")
cols=X_train.columns
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

print("Ended preprocessing")

print("Started training")
model= GaussianNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

print("Model Accuracy : {0:0.4f}".format(accuracy_score(y_test,y_pred)))
print("Model Accuracy : {0:0.4f}".format(accuracy_score(y_train,y_pred_train)))


print("Ended training")

print("Saving file innto pickle weights")
pickle.dump(model, open('model.pkl', 'wb'))

print("reading from pickle file")
pickled_model = pickle.load(open('model.pkl', 'rb'))
prediction = pickled_model.predict(X_test)
print(prediction)