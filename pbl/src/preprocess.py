# src/preprocess.py
import pandas as pd
import numpy as np

CRIME_COLS = [
    "Rape",
    "Kidnapping and Abduction",
    "Dowry Deaths",
    "Assault on women with intent to outrage her modesty",
    "Insult to modesty of Women",
    "Cruelty by Husband or his Relatives"
]

def load_and_clean(path):
    df = pd.read_csv(path)
    # Normalize column names
    df = df.rename(columns=lambda s: s.strip())
    # Ensure Year is int
    df['Year'] = df['Year'].astype(int)
    # Fill missing crime counts with 0 or interpolate as needed
    for c in CRIME_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
    # Standardize STATE/UT and DISTRICT strings
    df['STATE/UT'] = df['STATE/UT'].str.strip().str.title()
    df['DISTRICT'] = df['DISTRICT'].str.strip().str.title()
    return df

def aggregate_by_area(df, state, district=None):
    sub = df[df['STATE/UT']==state]
    if district:
        sub = sub[sub['DISTRICT']==district]
    # group by Year and sum crimes
    agg = sub.groupby('Year')[CRIME_COLS].sum().reset_index()
    return agg
