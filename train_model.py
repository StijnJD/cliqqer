import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURES = ["rssi", "actieve_kanalen", "rssi_variantie", "rssi_trend", "snelheid", "kanaal_freq"]

HULPDIENSTEN = {"politie", "ambulance", "brandweer"}

# ── Data laden ────────────────────────────────────────────────────────────────
df = pd.read_csv("tetra_simulatie.csv")

df["is_hulpdienst"] = df["label"].str.lower().isin(HULPDIENSTEN).astype(int)

X = df[FEATURES].values
y = df["is_hulpdienst"].values

# ── Train/test split 80/20 ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Model trainen ─────────────────────────────────────────────────────────────
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
])
model.fit(X_train, y_train)

# ── Evaluatie ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
                     index=["basisstation", "hulpdienst"],
                     columns=["pred_basisstation", "pred_hulpdienst"])
print(cm_df)

# ── Model opslaan ─────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "features": FEATURES}, f)

print("\nModel opgeslagen als model.pkl")


# ── Voorspelfunctie ───────────────────────────────────────────────────────────
def voorspel_hulpdienst(rssi: float, actieve_kanalen: int, rssi_variantie: float,
                        rssi_trend: float, snelheid: float, kanaal_freq: float,
                        model_path: str = "model.pkl") -> float:
    """
    Geeft P(hulpdienst) terug als een getal tussen 0 en 1.

    Parameters
    ----------
    rssi             : signaalsterkte in dBm
    actieve_kanalen  : aantal actieve kanalen
    rssi_variantie   : variantie van de RSSI
    rssi_trend       : trend in de RSSI (dBm/s)
    snelheid         : rijsnelheid in km/h
    kanaal_freq      : kanalfrequentie in MHz
    model_path       : pad naar het opgeslagen model (model.pkl)

    Returns
    -------
    float  kans dat het signaal afkomstig is van een hulpdienst (0.0 – 1.0)
    """
    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    clf = payload["model"]

    X_nieuw = np.array([[rssi, actieve_kanalen, rssi_variantie, rssi_trend, snelheid, kanaal_freq]])
    kans = clf.predict_proba(X_nieuw)[0][1]

    return round(float(kans), 4)


# ── Demonstratie ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── Voorbeeldvoorspelling ──")

    voorbeelden = [
        {"naam": "Basisstation",    "rssi": -65, "kanalen": 15, "var": 1.2,  "trend": 0.05, "snelheid": 0,   "freq": 380.0},
        {"naam": "Politievoertuig", "rssi": -88, "kanalen": 2,  "var": 14.5, "trend": 1.1,  "snelheid": 110, "freq": 391.5},
        {"naam": "Ambulance",       "rssi": -81, "kanalen": 2,  "var": 10.3, "trend": 0.6,  "snelheid": 75,  "freq": 393.0},
        {"naam": "Brandweer",       "rssi": -83, "kanalen": 3,  "var": 8.7,  "trend": 0.4,  "snelheid": 55,  "freq": 392.0},
    ]

    for vb in voorbeelden:
        p = voorspel_hulpdienst(vb["rssi"], vb["kanalen"], vb["var"], vb["trend"], vb["snelheid"], vb["freq"])
        print(f"  {vb['naam']:<20} P(hulpdienst) = {p:.4f}")
