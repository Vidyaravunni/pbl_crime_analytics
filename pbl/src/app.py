# src/app.py
import streamlit as st
import pandas as pd
from preprocess import load_and_clean, aggregate_by_area
from eda import plot_time_series, plot_top_crimes, plot_pie_composition, correlation_heatmap
from stats_utils import bootstrap_ci, two_sample_ttest
from ts_forecast import forecast_series
from similarity import build_feature_matrix, recommend_similar
import numpy as np
import os


DATA_PATH = "data/crime_data.csv"

@st.cache_data
def load_and_clean(path=DATA_PATH):
    # Load CSV
path = os.path.join(os.path.dirname(__file__), "..", "data", "crime_data.csv")
    df = pd.read_csv(path)

    # Normalize column names (strip spaces)
    df.columns = df.columns.str.strip()

    # Convert Year to int (if needed)
    if 'Year' in df.columns:
        df['Year'] = df['Year'].astype(int, errors='ignore')

    # 1️⃣ Remove rows where District == 'Total'
    df = df[df['DISTRICT'].str.strip().str.lower() != 'total']

    # 2️⃣ Normalize STATE/UT names (fix spacing and case)
    df['STATE/UT'] = df['STATE/UT'].str.strip().str.title()

    # 3️⃣ Fix known duplicates / inconsistent names
    df['STATE/UT'] = df['STATE/UT'].replace({
        'A & N Island': 'A&N Islands',
        'A & N Islands': 'A&N Islands',
        'A&N Island': 'A&N Islands',
        'A And N Island': 'A&N Islands',
        'A And N Islands': 'A&N Islands',
        'D & N Haveli': 'D&N Haveli',
        'D&N Haveli And Daman & Diu': 'D&N Haveli And Daman & Diu',
        'D & N Haveli And Daman & Diu': 'D&N Haveli And Daman & Diu'
    })

    # 4️⃣ Drop exact duplicate rows after cleanup
    df = df.drop_duplicates()

    return df



df = load_and_clean()

st.title("Crime Pattern Analysis — State/District Dashboard")

# ====== inputs
states = sorted(df['STATE/UT'].unique())
state = st.selectbox("Select State/UT", [""] + states)
districts = []
if state:
    districts = sorted(df[df['STATE/UT']==state]['DISTRICT'].unique())
district = st.selectbox("Select District (optional)", [""] + districts)

if st.button("Show Analysis"):
    if not state:
        st.error("Please select a state.")
    else:
        district_sel = district if district else None
        agg = aggregate_by_area(df, state, district_sel)
        if agg.empty:
            st.warning("No data for this selection.")
        else:
            st.subheader(f"Time series for {state}" + (f", {district_sel}" if district_sel else ""))
            st.plotly_chart(plot_time_series(agg, title="Crime counts over years"), use_container_width=True)

            st.plotly_chart(plot_top_crimes(agg), use_container_width=True)
            st.plotly_chart(plot_pie_composition(agg), use_container_width=True)
            st.plotly_chart(correlation_heatmap(agg), use_container_width=True)

            # Basic stats + bootstrap CI example for Rape
            rape_series = agg['Rape'].values
            mean_rape, ci_rape = bootstrap_ci(rape_series, np.mean, n_boot=2000)
            st.write("Rape — mean per year:", round(mean_rape,2))
            st.write("95% bootstrap CI:", (round(ci_rape[0],2), round(ci_rape[1],2)))

            # Forecast example (Rape)
            ts = agg.set_index('Year')['Rape']
            if len(ts) >= 3:
                pred_df, model = forecast_series(ts, order=(1,1,1), steps=5)
                # combine
                fig_df = pd.concat([ts.rename('observed'), pred_df['mean'].rename('forecast')], axis=0)
                st.line_chart(fig_df)
            else:
                st.info("Not enough years to forecast reliably (need >=3).")

            # Similarity / recommend
            matrix = build_feature_matrix(df)
            key = (state, district_sel if district_sel else "")
            # adjust key existence: our matrix index has (state,district)
            # find closest match if district empty: recommend by state-level sums
            # For simplicity, if district not provided, recommend other districts in same state
            try:
                # handle index type
                recs = recommend_similar((state, district_sel if district_sel else district_sel), matrix)
                st.subheader("Similar areas (by crime profile)")
                for r, s in recs:
                    st.write(r, round(s,3))
            except Exception:
                # fallback: recommend top districts in same state by totals
                st.info("Could not compute similarity for this selection; showing top districts in same state.")
                top = df[df['STATE/UT']==state].groupby('DISTRICT')[['Rape']].sum().sort_values('Rape', ascending=False).head(5)
                st.table(top)

st.sidebar.header("Project Modules")
st.sidebar.markdown("""
- EDA: time series, heatmaps  
- Stats: bootstrap CI, t-tests  
- Forecasting: ARIMA  
- Similarity & recommender  
- Optional: NLP, network analysis  
""")




