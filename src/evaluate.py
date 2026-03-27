from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))