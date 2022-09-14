from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn import metrics
from main import X_train, X_test, Y_train, Y_test
from sklearn.metrics import confusion_matrix
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
y_pred = gaussian.predict(X_test)
print('confusion_matrix = : ', confusion_matrix(Y_test, y_pred))
print('mean square error  = : ', mean_squared_error(Y_test, y_pred))
print('Classification report:\n', metrics.classification_report(Y_test, y_pred))
print("Naive bayes Accuracy: ", accuracy_score(Y_test, y_pred))