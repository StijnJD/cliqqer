import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N = 1000
start_time = datetime(2026, 3, 24, 8, 0, 0)

ALLE_KANALEN = np.round(np.arange(385.000, 390.001, 0.025), 3)
POLITIE_KANALEN    = ALLE_KANALEN[ALLE_KANALEN > 387.5]
AMBULANCE_KANALEN  = ALLE_KANALEN[ALLE_KANALEN < 387.5]
BRANDWEER_KANALEN  = ALLE_KANALEN[(ALLE_KANALEN >= 386.0) & (ALLE_KANALEN <= 388.0)]

klassen = {
    "basisstation": 250,
    "politie":      250,
    "ambulance":    250,
    "brandweer":    250,
}

records = []

for label, count in klassen.items():
    for i in range(count):
        timestamp = start_time + timedelta(seconds=np.random.randint(0, 86400))

        if label == "basisstation":
            rssi            = np.random.normal(-65, 3)
            actieve_kanalen = np.random.randint(10, 20)
            rssi_variantie  = np.random.uniform(0.5, 2.5)
            rssi_trend      = np.random.uniform(-0.2, 0.2)
            snelheid        = 0.0
            gps_lat         = np.random.uniform(52.0, 53.5)
            gps_lon         = np.random.uniform(4.0, 6.5)
            kanaal_freq     = float(np.random.choice(ALLE_KANALEN))

        elif label == "politie":
            rssi            = np.random.normal(-85, 10)
            actieve_kanalen = np.random.randint(1, 5)
            rssi_variantie  = np.random.uniform(8, 20)
            rssi_trend      = np.random.uniform(-1.5, 1.5)
            snelheid        = np.random.uniform(0, 160)
            gps_lat         = np.random.uniform(52.0, 53.5)
            gps_lon         = np.random.uniform(4.0, 6.5)
            kanaal_freq     = float(np.random.choice(POLITIE_KANALEN))

        elif label == "ambulance":
            rssi            = np.random.normal(-80, 8)
            actieve_kanalen = np.random.randint(1, 4)
            rssi_variantie  = np.random.uniform(6, 18)
            rssi_trend      = np.random.uniform(-1.0, 1.0)
            snelheid        = np.random.uniform(0, 120)
            gps_lat         = np.random.uniform(52.0, 53.5)
            gps_lon         = np.random.uniform(4.0, 6.5)
            kanaal_freq     = float(np.random.choice(AMBULANCE_KANALEN))

        elif label == "brandweer":
            rssi            = np.random.normal(-82, 9)
            actieve_kanalen = np.random.randint(2, 6)
            rssi_variantie  = np.random.uniform(5, 15)
            rssi_trend      = np.random.uniform(-0.8, 0.8)
            snelheid        = np.random.uniform(0, 90)
            gps_lat         = np.random.uniform(52.0, 53.5)
            gps_lon         = np.random.uniform(4.0, 6.5)
            kanaal_freq     = float(np.random.choice(BRANDWEER_KANALEN))

        records.append({
            "timestamp":       timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "rssi":            round(float(rssi), 2),
            "actieve_kanalen": int(actieve_kanalen),
            "rssi_variantie":  round(float(rssi_variantie), 3),
            "rssi_trend":      round(float(rssi_trend), 4),
            "gps_lat":         round(float(gps_lat), 6),
            "gps_lon":         round(float(gps_lon), 6),
            "snelheid":        round(float(snelheid), 1),
            "kanaal_freq":     round(float(kanaal_freq), 3),
            "label":           label,
        })

df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)

output_path = "tetra_simulatie.csv"
df.to_csv(output_path, index=False)

print(f"Dataset opgeslagen: {output_path}")
print(f"Totaal records : {len(df)}")
print(f"\nVerdeling labels:")
print(df["label"].value_counts())
print(f"\nVoorbeeld (eerste 5 rijen):")
print(df.head())
