from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn import metrics
from main import X_train, X_test, Y_train, Y_test
model_tree = DecisionTreeClassifier(criterion='gini', max_depth=2)
model_tree.fit(X_train, Y_train)
y_pred_tree = model_tree.predict(X_test)
accuracy_tree = metrics.accuracy_score(Y_test, y_pred_tree)
print('accuracy test: ', accuracy_score(Y_test, y_pred_tree))
print('accuracy in percentage of decision tree model: ', int(accuracy_tree*100), '%')
print('confusion matrix:\n', metrics.confusion_matrix(Y_test, y_pred_tree))
print('Classification report:\n', metrics.classification_report(Y_test, y_pred_tree))
print('mean square error  = : ',mean_squared_error(Y_test,y_pred_tree))