from logistic_regression import *
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_bow, ytrain)
prediction = rf.predict(xvalid_bow)
# validation score
print(f'F1 score for RandomForest BOW features {f1_score(yvalid, prediction)}')

# create submission file
test_pred = rf.predict(test_bow)
test['label'] = test_pred
submission = test[['id','label']]
submission.to_csv('sub_rf_bow.csv', index=False)

# predict o TF-IDF features
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_tfidf, ytrain)
prediction = rf.predict(xvalid_tfidf)
print(f'F1 score for RandomForest BOW features {f1_score(yvalid, prediction)}')

