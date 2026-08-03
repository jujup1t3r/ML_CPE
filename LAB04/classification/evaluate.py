import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def evaluate_and_save(y_test, y_pred, output_cm="outputs/02_confusion_matrix.png", output_csv="outputs/predictions.csv"):
    """
    Generate Confusion Matrix plot and save predictions.csv
    """
    # 1. Print Classification Report
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    # 2. Save Confusion Matrix Plot
    cm_percentage = confusion_matrix(y_test, y_pred, normalize='true') * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=["No Disease", "Disease"])
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.2f')

    plt.title("Confusion Matrix (%)")
    plt.savefig(output_cm, bbox_inches='tight')
    plt.close()

    # 3. Save Predictions CSV
    df_res = pd.DataFrame({
        'Actual_Heart_Disease': y_test,
        'Predicted_Heart_Disease': y_pred
    })
    df_res.to_csv(output_csv, index=False)
    print(f"Predictions saved to '{output_csv}'")
    print(f"Confusion Matrix saved to '{output_cm}'")