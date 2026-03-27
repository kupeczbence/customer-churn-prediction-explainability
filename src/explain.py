import shap
import matplotlib.pyplot as plt
import os

def explain_model(model, X):
    # create reports folder if not exists
    os.makedirs("reports", exist_ok=True)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("reports/shap_summary.png", bbox_inches="tight")
    plt.close()

    print("SHAP summary saved to reports/shap_summary.png")