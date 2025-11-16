import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1) Load your merged CSV
df = pd.read_csv("반도체 데이터 수집.csv")   

# 2) Strip whitespace from column names (fix 'Export ' etc.)
df.columns = df.columns.str.strip()

print(df.columns)

# Parse Date column and sort
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

numeric_cols = [
    "Export",
    "DRAM",
    "Exchange Rate",                
    "CAPEX",
    "OECD CLI",
    "U.S. ISM Manufacturing New Orders Index"   
]

for col in numeric_cols:
    df[col] = (
        df[col]
        .astype(str)          # ensure string
        .str.replace(",", "", regex=False)  # remove commas
        .replace("", pd.NA)   # empty → NaN
        .astype(float)        # convert to float
    )

# Ignore earlier value
df = df[df["Date"] >= "2011-03-01"].copy()
df = df.sort_values("Date").reset_index(drop=True)

#Ignore Global PMI first
df = df.drop(columns=["Global Manufacturing PMI"], errors="ignore")

# Set Date as index (optional but convenient)
df = df.set_index("Date")

feature_cols = [
    "Export",
    "DRAM",
    "Exchange Rate",
    "CAPEX",
    "OECD CLI",
    "U.S. ISM Manufacturing New Orders Index"
]

scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df[feature_cols]),
    index=df.index,
    columns=feature_cols,
)

df_scaled.head()

# Prepare sequence for CT-VAE
L = 24  # lookback length

values = df_scaled.values  # shape: (T, D)

X_seq = []
for i in range(len(values) - L + 1):
    X_seq.append(values[i:i+L])

X_seq = np.stack(X_seq)  # shape: (N_sequences, L, D)

print(X_seq.shape)

