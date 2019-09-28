from xgboost import XGBClassifier
from logistic_regression import *

# BOW features
xgb_model = XGBClassifier(max_depth=6, n_estimators=1000).fit(xtrain_bow, ytrain)
prediction = xgb_model.predict(xvalid_bow)
print(f'F1 score for XGBoost BOW features {f1_score(yvalid, prediction)}')


test_pred = xgb_model.predict(test_bow)
test['label'] = test_pred
submission = test[['id','label']]
submission.to_csv('sub_xgb_bow.csv', index=False)
