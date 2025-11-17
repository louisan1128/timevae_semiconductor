# ============================================================
#   CT-VAE / TCN 전처리 FULL PIPELINE
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1) CSV 로드
# ------------------------------------------------------------
df = pd.read_csv(
    r"data.csv"
)

# 날짜 변환
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# ------------------------------------------------------------
# 2) 숫자로 변환 (Date 제외)
# ------------------------------------------------------------

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ------------------------------------------------------------
# 3) CAPEX (already 0 except quarterly) → log transform
#    (CAPEX는 분기별로만 값이 있고 나머지는 0이라 log1p 적용)
# ------------------------------------------------------------
df['CAPEX'] = np.log1p(df['CAPEX'])

# ------------------------------------------------------------
# 4) Condition features missing → 0
#    (PMI, CLI, ISM은 중간부터 있어서 0으로 채우는 것이 정석)
#    없는 데이터들은 일단 0으로 채움
# ------------------------------------------------------------
condition_raw_cols = [
    'Exchange Rate',
    'CAPEX',
    'Global Manufacturing PMI',
    'OECD CLI',
    'U.S. ISM Manufacturing New Orders Index'
]

df[condition_raw_cols] = df[condition_raw_cols].fillna(0)

# ------------------------------------------------------------
# 5) StandardScaler (Z-score)
# ------------------------------------------------------------
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    index=df.index,
    columns=df.columns
)

# ------------------------------------------------------------
# 6) Sliding Window 생성
# ------------------------------------------------------------

L = 36      # 입력 길이
H = 12      # 예측 길이

def create_dataset(df, L, H, cond_cols):
    X, Y, C = [], [], []
    total_len = len(df)

    for start in range(total_len - L - H):
        end_x = start + L
        end_y = end_x + H

        # 시계열 입력 X (L, D)
        x = df.iloc[start:end_x].values

        # Y future (H, D)
        y = df.iloc[end_x:end_y].values

        # Condition c (raw condition features)
        # window 마지막 시점에서 가져옴
        c = df.iloc[end_x - 1][cond_cols].values

        X.append(x)
        Y.append(y)
        C.append(c)

    return np.array(X), np.array(Y), np.array(C)

X, Y, C = create_dataset(df_scaled, L, H, condition_raw_cols)

# ------------------------------------------------------------
# 8) 결과 shape 출력
# ------------------------------------------------------------

print("X shape:", X.shape)   # (B, 36, 7)
print("Y shape:", Y.shape)   # (B, 12, 7)
print("C shape:", C.shape)   # (B, 5) → 이후 ConditionLayer로 (B, 10)







# ------------------------------------------------------------
# 0) training용 함수 버전 (import용)
# ------------------------------------------------------------
def preprocess(csv_path="data.csv"):
    df = pd.read_csv(csv_path)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['CAPEX'] = np.log1p(df['CAPEX'])

    df[condition_raw_cols] = df[condition_raw_cols].fillna(0)

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )

    X, Y, C = create_dataset(df_scaled, L, H, condition_raw_cols)

    ##float 32로 지정 (충돌 방지)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    C = C.astype(np.float32)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("C shape:", C.shape)

    return X, Y, C
