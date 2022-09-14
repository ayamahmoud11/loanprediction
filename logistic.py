# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from main import X_train, X_test, Y_train, Y_test
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn import metrics

log_reg = LogisticRegression(solver='liblinear')
log_reg.fit(X_train, Y_train)
y_pred = log_reg.predict(X_test)
print('logistic regression accuracy : ', accuracy_score(Y_test, y_pred))
print('logistic regression recall : ', recall_score(Y_test, y_pred))
print('confusion_matrix = : ', confusion_matrix(Y_test, y_pred))
print('mean square error  = : ',mean_squared_error(Y_test,y_pred))
print('Classification report:\n', metrics.classification_report(Y_test, y_pred))