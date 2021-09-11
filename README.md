# SVM Practice (Udacity: Intro to Machine Learning)
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

``` python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
import math

features_train = features_train[:math.ceil(len(features_train)/100)]
labels_train = labels_train[:math.ceil(len(labels_train)/100)]

clf = SVC(kernel='rbf',C=10000)
t0 = time()
clf.fit(features_train,labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

print(accuracy_score(labels_test,pred))

```
