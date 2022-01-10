import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import plotly.express as px

from google.colab import files
dataset = files.upload()

dataset = pd.read_csv("cancer_classification.csv"); dataset

dataset['diagnosis'] = [1 if x=="M" else 0 for x in dataset['diagnosis']]; dataset
dataset["diagnosis"].value_counts().plot(kind='bar')

dataset['diagnosis'].value_counts()
dataset.isnull().sum()

corr = dataset.corr()
plt.subplots(figsize=(15,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))

dataset=dataset.drop(["area_mean","perimeter_mean","area_se","perimeter_se","area_worst","perimeter_worst","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst","concave points_mean","concave points_se","compactness_mean","compactness_se","id","Unnamed: 32"],axis=1)

X = dataset.drop(['diagnosis'], axis=1); X

Y = dataset['diagnosis']; Y

from sklearn.preprocessing import StandardScaler
X_standard = StandardScaler().fit_transform(X); X_standard

from sklearn.datasets import make_classification
from matplotlib import pyplot

from sklearn.tree import DecisionTreeClassifier
model_imp = DecisionTreeClassifier(random_state=12345)
model_imp.fit(X_standard,Y)
importance = model_imp.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

X_standard=pd.DataFrame(X_standard);X_standard
X_standard=X_standard.drop([4,5,7],axis=1)

from  sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_standard, Y, test_size=0.3, random_state= 22)

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
class_tree=DecisionTreeClassifier(random_state=22)
param = {'max_depth':range(1, 6), 'min_samples_split':range(10,20), 'min_samples_leaf':range(2,15)} 
grid = GridSearchCV(class_tree, param, cv=5)
grid.fit(X_train, Y_train)
print(grid.best_params_)

from sklearn import tree
mod_tree=DecisionTreeClassifier(max_depth=3,min_samples_split=10,min_samples_leaf=6,random_state=22)
mod_tree.fit(X_train, Y_train)
text_representation = tree.export_text(mod_tree)
print(text_representation)

fig1 = plt.figure(figsize=(25,20))
_ = tree.plot_tree(mod_tree,feature_names=list(X.columns), class_names=['Benign', 'Malignant'],filled=True)

mod_tree.fit(X_train, Y_train)#Test set
y_pred = mod_tree.predict(X_test)
print(classification_report(Y_test, y_pred))
