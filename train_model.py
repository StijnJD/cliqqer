import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

FEATURES = ["rssi", "actieve_kanalen", "rssi_variantie", "rssi_trend", "snelheid"]

# ── Data laden ────────────────────────────────────────────────────────────────
df = pd.read_csv("tetra_simulatie.csv")

X = df[FEATURES].values
y = df["label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

# ── Train/test split 80/20 ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ── Model trainen ─────────────────────────────────────────────────────────────
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
])
model.fit(X_train, y_train)

# ── Evaluatie ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print(cm_df)

# ── Model opslaan ─────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "label_encoder": le, "features": FEATURES}, f)

print("\nModel opgeslagen als model.pkl")


# ── Voorspelfunctie ───────────────────────────────────────────────────────────
def voorspel_kansen(rssi: float, actieve_kanalen: int, rssi_variantie: float,
                    rssi_trend: float, snelheid: float,
                    model_path: str = "model.pkl") -> dict:
    """
    Geeft de kans per klasse terug als dictionary.

    Parameters
    ----------
    rssi             : signaalsterkte in dBm
    actieve_kanalen  : aantal actieve kanalen
    rssi_variantie   : variantie van de RSSI
    rssi_trend       : trend in de RSSI (dBm/s)
    snelheid         : rijsnelheid in km/h
    model_path       : pad naar het opgeslagen model (model.pkl)

    Returns
    -------
    dict  {klasse: kans, ...}  gesorteerd op aflopende kans
    """
    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    clf = payload["model"]
    enc = payload["label_encoder"]

    X_nieuw = np.array([[rssi, actieve_kanalen, rssi_variantie, rssi_trend, snelheid]])
    kansen  = clf.predict_proba(X_nieuw)[0]

    resultaat = {klasse: round(float(kans), 4)
                 for klasse, kans in zip(enc.classes_, kansen)}

    return dict(sorted(resultaat.items(), key=lambda x: x[1], reverse=True))


# ── Demonstratie ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── Voorbeeldvoorspelling ──")

    voorbeelden = [
        {"naam": "Basisstation",  "rssi": -65, "kanalen": 15, "var": 1.2,  "trend": 0.05, "snelheid": 0},
        {"naam": "Politievoertuig", "rssi": -88, "kanalen": 2, "var": 14.5, "trend": 1.1,  "snelheid": 110},
        {"naam": "Ambulance",     "rssi": -81, "kanalen": 2, "var": 10.3, "trend": 0.6,  "snelheid": 75},
        {"naam": "Brandweer",     "rssi": -83, "kanalen": 3, "var": 8.7,  "trend": 0.4,  "snelheid": 55},
    ]

    for vb in voorbeelden:
        kansen = voorspel_kansen(vb["rssi"], vb["kanalen"], vb["var"], vb["trend"], vb["snelheid"])
        print(f"\n{vb['naam']}:")
        for klasse, kans in kansen.items():
            print(f"  {klasse:<15} {kans:.2%}")
