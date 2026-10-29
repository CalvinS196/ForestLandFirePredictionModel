from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

validation_prediction = model.predict(X_validation)
test_prediction = model.predict(X_test)

print("Validation")
print(classification_report(y_validation, validation_prediction, zero_division=0))

print("Test")
print(classification_report(y_test, test_prediction, zero_division=0))

test_probability = model.predict_proba(X_test)[:, 1]
print("Test ROC-AUC:", roc_auc_score(y_test, test_probability))