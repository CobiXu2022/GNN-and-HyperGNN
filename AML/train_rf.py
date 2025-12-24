# train_rf.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, 
                            confusion_matrix, classification_report)
from dataset_rf import AMLtoRF  
import joblib
def main():
    data_processor = AMLtoRF(root='/workspace/data')
    
    X_train, X_test, y_train, y_test = data_processor.get_train_test_split()
    
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}')
    print(f'AUROC: {auroc:.4f}')
    print(f'F1-Score: {f1:.4f}')
    recall = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    print(f'Recall (TPR): {recall:.4f}')
    print('\nConfusion Matrix:')
    print(f'[[TN FP]  [{cm[0,0]} {cm[0,1]}]\n [FN TP]]  [{cm[1,0]} {cm[1,1]}]]')
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    print(f'Specificity (TNR): {specificity:.4f}')

    joblib.dump(rf_model, 'aml_rf_model.pkl')
    print("\nModel saved to aml_rf_model.pkl")

if __name__ == "__main__":
    main()