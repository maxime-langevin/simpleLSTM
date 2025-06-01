import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from tqdm import tqdm


def smiles_to_ecfp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def compute_tanimoto(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def main():
    # --- Load and prepare dataset ---
    df = pd.read_csv("data/jak2_data.csv")
    df = df.dropna(subset=["SMILES", "pIC50"])
    df["Active"] = df["pIC50"] > 7.0

    fps, valid_smiles, labels = [], [], []

    for _, row in df.iterrows():
        fp = smiles_to_ecfp(row["SMILES"])
        if fp is not None:
            fps.append(fp)
            valid_smiles.append(row["SMILES"])
            labels.append(row["Active"])

    # Convert to feature matrix
    X = []
    for fp in fps:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        X.append(arr)
    X = np.array(X)
    y = np.array(labels)

    # --- Train-test split ---
    X_train, X_test, y_train, y_test, fps_train, fps_test = train_test_split(
        X, y, fps, test_size=0.2, random_state=42
    )

    # --- Train classifier ---
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = y_pred_proba > 0.5

    # --- Evaluation metrics ---
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC: {auc:.3f}")

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_auc_curve.png")

    # --- Error correlation vs Tanimoto similarity ---
    bin_edges = np.linspace(0, 1, 6)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    errors = (y_pred != y_test).astype(int)
    correlations = []

    print("Computing error correlation vs Tanimoto similarity...")

    for i in range(len(bin_edges) - 1):
        errs_j = []
        errs_k = []
        for j in tqdm(range(len(fps_test))):
            for k in range(j + 1, len(fps_test)):
                sim = compute_tanimoto(fps_test[j], fps_test[k])
                if bin_edges[i] <= sim < bin_edges[i + 1]:
                    errs_j.append(errors[j])
                    errs_k.append(errors[k])
        if len(errs_j) > 1:
            corr, _ = pearsonr(errs_j, errs_k)
            correlations.append(corr)
        else:
            correlations.append(np.nan)

    # --- Plot error correlation ---
    plt.figure()
    plt.plot(bin_centers, correlations, marker="o")
    plt.xlabel("Tanimoto Similarity")
    plt.ylabel("Error Correlation")
    plt.grid(True)
    plt.savefig("error_corr_vs_similarity.png")
    plt.show()


if __name__ == "__main__":
    main()
