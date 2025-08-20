from sklearn.metrics import classification_report, confusion_matrix


def compute_metrics(y_true, y_pred) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        
        # "full_report": report,
        # "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

def full_report(y_true, y_pred) -> dict:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    rep["confusion_matrix"] = cm
    return rep