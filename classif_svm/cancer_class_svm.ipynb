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
