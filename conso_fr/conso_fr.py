import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from q3dfm.dfm import *

ggtrends = pd.read_csv(
    "/Users/maximecoulet/CSENSAE/q3dfm/conso_fr/data/all_deseasonalized_trends.csv"
)
threshold = 0.01 * len(ggtrends)

# Dropping columns with more than 80% NaN or zeros
ggtrends_cleaned = (
    ggtrends[1:]
    .drop(
        columns=[
            col
            for col in ggtrends.columns
            if ggtrends[col].isna().sum() > threshold
            or (ggtrends[col] == 0).sum() > threshold
        ]
    )
    .set_index("date")
)

ggtrends_transformed = np.log(ggtrends_cleaned + 100).diff(1)

scaler = StandardScaler()
ggtrends_standardized = (
    ggtrends_transformed.copy()
)  # Créer une copie pour ne pas affecter df_cleaned
for col in ggtrends_standardized.columns:
    ggtrends_standardized[col] = scaler.fit_transform(
        ggtrends_standardized[[col]]
    )

np_ggtrends_standardized = ggtrends_standardized.iloc[1:, :].to_numpy()
l = np_ggtrends_standardized.shape[1]

result = dfm(
    np_ggtrends_standardized,
    np_ggtrends_standardized,
    m=8,
    p=1,
    frq=["w"] * l,
    isdiff=[True] * l,
)
