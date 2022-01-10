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

from sklearn import svm
parameters = {'kernel':('linear', 'rbf',"poly","sigmoid"), 'C':range(1, 10)}
svc = svm.SVC(random_state=22)
grid_svm = GridSearchCV(svc, parameters,cv=5)
grid_svm.fit(X_train, Y_train)
print(grid_svm.best_params_)

Support_class = svm.SVC(C=1,kernel="rbf",random_state=22)
Support_class.fit(X_train, Y_train)
y_train_pred = Support_class.predict(X_train)
print(classification_report(Y_train, y_train_pred))


y_pred = Support_class.predict(X_test)
print(classification_report(Y_test, y_pred))

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_svm=pca.fit_transform(X_standard);pca_svm
print(pca.explained_variance_ratio_)
pca_svm=pd.DataFrame(pca_svm);pca_svm

plt.scatter(pca_svm[0],pca_svm[1],c=Y)

X_PCA_train,X_PCA_test=train_test_split(pca_svm,test_size=0.3,random_state=22)

parameters1 = {'kernel':('linear', 'rbf',"poly","sigmoid"), 'C':range(1, 10)}
svc1 = svm.SVC(random_state=22)
grid_svm1= GridSearchCV(svc1, parameters,cv=5)
grid_svm1.fit(X_PCA_train, Y_train)
print(grid_svm1.best_params_)

Support_class1= svm.SVC(C=2,kernel="rbf",random_state=22)
Support_class1.fit(X_PCA_train, Y_train)

y_train_pred1= Support_class1.predict(X_PCA_train)
print(classification_report(Y_train, y_train_pred1))
y_pred1= Support_class1.predict(X_PCA_test)
print(classification_report(Y_test, y_pred1))

plt.scatter(X_PCA_train[0], X_PCA_train[1])
plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(np.array(X_PCA_train), np.array(Y_train), clf=Support_class1, legend=2)
plt.show()
support_vectors = Support_class1.support_vectors_
