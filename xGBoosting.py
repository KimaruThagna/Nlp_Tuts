from xgboost import XGBClassifier
import xgboost as xgb
from logistic_regression import *

# BOW features
xgb_model = XGBClassifier(max_depth=6, n_estimators=1000).fit(xtrain_bow, ytrain)
prediction = xgb_model.predict(xvalid_bow)
print(f'F1 score for XGBoost BOW features {f1_score(yvalid, prediction)}')


test_pred = xgb_model.predict(test_bow)
test['label'] = test_pred
submission = test[['id','label']]
submission.to_csv('sub_xgb_bow.csv', index=False)

# TF-IDF features
xgb = XGBClassifier(max_depth=6, n_estimators=1000).fit(xtrain_tfidf, ytrain)
prediction = xgb.predict(xvalid_tfidf)
print(f'F1 score for XGBoost TF-IDF features {f1_score(yvalid, prediction)}')

# Word2Vec features
xgb = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3).fit(xtrain_w2v, ytrain)
prediction = xgb.predict(xvalid_w2v)
print(f'F1 score for XGBoost Word2Vec features {f1_score(yvalid, prediction)}')

# doc2Vec
xgb = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3).fit(xtrain_d2v, ytrain)
prediction = xgb.predict(xvalid_d2v)
print(f'F1 score for XGBoost Doc2Vec features {f1_score(yvalid, prediction)}')


'''
XGBOOST + WORD2VEC  PARAMETER TUNING
'''
#using D matrices. Can hold both features and target variable
dtrain = xgb.DMatrix(xtrain_w2v, label=ytrain)
dvalid = xgb.DMatrix(xvalid_w2v, label=yvalid)
dtest = xgb.DMatrix(test_w2v)
# Parameters that we are going to tune
params = {
    'objective':'binary:logistic',
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1
 }

# define evaluation metric
def custom_eval(preds, dtrain):
    labels = dtrain.get_label().astype(np.int)
    preds = (preds >= 0.3).astype(np.int)
    return [('f1_score', f1_score(labels, preds))]


#Tuning max_depth and min_child_weight

gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(6, 10)
    for min_child_weight in range(5, 8)
]
max_f1 = 0.  # initializing with 0
best_params = None
cv_results = {}
for max_depth, min_child_weight in gridsearch_params:
    print(f'CV with max_depth={max_depth}, min_child_weight={min_child_weight}')
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

    # Cross-validation
    cv_results = xgb.cv(params,
                        dtrain,
                        feval=custom_eval,
                        num_boost_round=200,
                        maximize=True,
                        seed=16,
                        nfold=5,
                        early_stopping_rounds=10
                        )
# Finding best F1 Score

mean_f1 = cv_results['test-f1_score-mean'].max()
boost_rounds = cv_results['test-f1_score-mean'].argmax()
print(f'\tF1 Score {mean_f1} for {boost_rounds}')
if mean_f1 > max_f1:
    max_f1 = mean_f1
    best_params = (max_depth, min_child_weight)

print(f'Best params: {best_params[0]}, {best_params[1]}, F1 Score: { max_f1}')

#Updating max_depth and min_child_weight parameters.

params['max_depth'] = best_params[0]
params['min_child_weight'] = best_params[1]

#Tuning subsample and colsample

gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(5,10)]
    for colsample in [i/10. for i in range(5,10)] ]
max_f1 = 0.
best_params = None
for subsample, colsample in gridsearch_params:
    print(f'CV with subsample={subsample}, colsample={colsample}')
     # Update our parameters
    params['colsample'] = colsample
    params['subsample'] = subsample
    cv_results = xgb.cv(
        params,
        dtrain,
        feval= custom_eval,
        num_boost_round=200,
        maximize=True,
        seed=16,
        nfold=5,
        early_stopping_rounds=10
    )
     # Finding best F1 Score
    mean_f1 = cv_results['test-f1_score-mean'].max()
    boost_rounds = cv_results['test-f1_score-mean'].argmax() # position of the maximum element
    print(f'\tF1 Score {mean_f1} for {boost_rounds}')
    if mean_f1 > max_f1:
        max_f1 = mean_f1
        best_params = (subsample, colsample)

print(f'Best params: {best_params[0]}, {best_params[1]}, F1 Score: { max_f1}')

params['subsample'] = best_params[0]
params['colsample_bytree'] = best_params[1]
