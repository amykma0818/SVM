# SVM Practice (Udacity: Intro to Machine Learning)
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
