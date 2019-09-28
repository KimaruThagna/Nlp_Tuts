from sklearn import svm
from sklearn.metrics import f1_score
from logistic_regression import xtrain_bow, ytrain, xvalid_bow, test_bow, yvalid, np, test

svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_bow, ytrain)
prediction = svc.predict_proba(xvalid_bow)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
print(f'F1 score for SVM BOW features {f1_score(yvalid, prediction_int)}')

test_pred = svc.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('sub_svm_bow.csv', index=False)