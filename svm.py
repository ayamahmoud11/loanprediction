from sklearn.svm import SVC
from main import X_test, X_train, Y_train, Y_test
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, plot_confusion_matrix
from sklearn import metrics

model: SVC = svm.SVC(kernel='linear', C=100)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print("SVM Accuracy: ", accuracy_score(Y_test, Y_pred))
print("SVM Precision: ", precision_score(Y_test, Y_pred))
print("SVM Score: ", model.score(X_train, Y_train))
print("SVM Recall: ", recall_score(Y_test, Y_pred))

print("classification report: ", metrics.classification_report(Y_test, Y_pred))
print("confusion matrix:", metrics.confusion_matrix(Y_test, Y_pred))
print("mean squared error:", mean_squared_error(Y_test, Y_pred))

