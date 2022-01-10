# cancer_class
classification project with medical data using decision tree.
The data have been downloaded from the website kaggle.com and the analysis was coducted using Google colab.
The datadet had 569 observations and the target variable "diagnosis", which had two outputs "M" and "B". WHere "M" stands for malignant and "B" for benign.
These ooutputs have been then changed in a binary encoding (1 for "M" and 0 for "B") in order to compute the correlation between this variable and all the predictors. 

Subsequently, missing data have ben checked and detected for the variable "Unnamed: 32". Becasue this variable had only missing data, it has been removed. Furthermore, the variable "id" has been removed because it is not useful ofr the analysis. Once removed those variables, a correlation matrix has been computed in order to make feature selection. 

- Variables related to perimeter and area have been removed because they are highly correlated with the variable "radius"; 
- variables that end with "worst" have been removed because they are highly correlated with the variables that end with "mean";
- variables related to "concave points" and "compactness" have ben removed because they are highly correlated with "concavity".
Subsequently, the variables have been standardized because they have different units of measure. Feature selectiion has been then carried out through feature importance score computation. 

Once completed feature selection, the dataset has been splitted into training set (70%) and test set (30%). To identify the best hyper-parameter configuration for the decision tree gridsearch cross-validation. Once gained the best configuration, the tree obtained had 8 terminal nodes, which have low impurity values. In addition, the model has been evaluated on both the training set and the test set through F-1 score because the classes are not balanced. The model has high scores on both sets, therefore it has good classification performances and does not show overfitting. 
