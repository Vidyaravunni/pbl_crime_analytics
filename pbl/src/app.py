# pbl/src/app.py

import streamlit as st
import pandas as pd
import numpy as np
import os

from preprocess import aggregate_by_area
from eda import plot_time_series, plot_top_crimes, plot_pie_composition, correlation_heatmap
from stats_utils import bootstrap_ci
from ts_forecast import forecast_series
from similarity import build_feature_matrix, recommend_similar


# =====================
# DATA LOADING
# =====================
@st.cache_data
def load_and_clean():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "crime_data.csv")
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()
    df['Year'] = df['Year'].astype(int, errors='ignore')

    df = df[df['DISTRICT'].str.strip().str.lower() != 'total']
    df['STATE/UT'] = df['STATE/UT'].str.strip().str.title()

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

    return df.drop_duplicates()


df = load_and_clean()


# =====================
# SESSION STATE
# =====================
if "show_report" not in st.session_state:
    st.session_state.show_report = False


# =====================
# UI HEADER
# =====================
st.title("Crime Pattern Analysis — State/District Dashboard")


# =====================
# 1. USER INPUT SECTION
# =====================
states = sorted(df['STATE/UT'].unique())
state = st.selectbox("Select State/UT", [""] + states)

districts = []
if state:
    districts = sorted(df[df['STATE/UT'] == state]['DISTRICT'].unique())

district = st.selectbox("Select District (optional)", [""] + districts)


# =====================
# REPORT CRIME BUTTON
# =====================
if state and district:
    if st.button("📝 Report Crime"):
        st.session_state.show_report = not st.session_state.show_report


# =====================
# 2. USER-REPORTED CRIME INPUT
# =====================
if st.session_state.show_report and state and district:

    st.subheader("USER-REPORTED CRIME INPUT")

    crime_columns = [
        "Rape",
        "Kidnapping and Abduction",
        "Dowry Deaths",
        "Assault on women with intent to outrage her modesty",
        "Insult to modesty of Women",
        "Cruelty by Husband or his Relatives"
    ]

    user_inputs = {}

    for col in crime_columns:
        user_inputs[col] = st.number_input(
            f"Enter number of incidents for '{col}'",
            min_value=0,
            value=0,
            key=col
        )

    if st.button("✅ Submit Report"):
        for col in crime_columns:
            df.loc[
                (df['STATE/UT'] == state) &
                (df['DISTRICT'] == district),
                col
            ] += user_inputs[col]

        st.success("✅ Crime report successfully recorded!")
        st.session_state.show_report = False


# =====================
# 3. ANALYSIS
# =====================
if st.button("Show Analysis"):

    if not state:
        st.error("Please select a state.")
    else:
        district_sel = district if district else None
        agg = aggregate_by_area(df, state, district_sel)

        if agg.empty:
            st.warning("No data for this selection.")
        else:
            st.subheader(f"Crime Analysis for {state}" + (f" - {district_sel}" if district_sel else ""))

            # --------------------
            # TIME SERIES
            # --------------------
            st.plotly_chart(plot_time_series(agg, title="Year-wise Crime Trend"), use_container_width=True)

            trend = agg.iloc[:, 1:].sum(axis=1)
            if trend.iloc[-1] > trend.iloc[0]:
                st.info("📈 Crime trend is increasing over the years in this region.")
            else:
                st.info("📉 Crime trend is decreasing or stable over the years in this region.")


            # --------------------
            # TOP CRIMES
            # --------------------
            st.plotly_chart(plot_top_crimes(agg), use_container_width=True)

            crime_totals = agg.iloc[:, 1:].sum()
            top_crime = crime_totals.idxmax()
            st.warning(f"🚨 The most common crime in this region is: **{top_crime}**")


            # --------------------
            # PIE CHART
            # --------------------
            st.plotly_chart(plot_pie_composition(agg), use_container_width=True)

            top_share = (crime_totals.max() / crime_totals.sum()) * 100
            st.info(f"📊 **{top_crime}** accounts for about **{top_share:.1f}%** of all major crimes here.")


            # --------------------
            # HEATMAP
            # --------------------
            st.plotly_chart(correlation_heatmap(agg), use_container_width=True)

            st.write(
                "🧩 The heatmap shows which crimes increase together. "
                "Darker colors mean crimes are related (when one rises, the other also tends to rise)."
            )


            # --------------------
            # BOOTSTRAP MEAN
            # --------------------
            rape_series = agg["Rape"].values
            mean_rape, ci_rape = bootstrap_ci(rape_series, np.mean, n_boot=2000)

            st.write(f"📌 Average reported rape cases per year: **{round(mean_rape, 2)}**")
            st.write(f"📌 Expected yearly range: between **{round(ci_rape[0], 2)} and {round(ci_rape[1], 2)}** cases")


            # --------------------
            # FORECAST
            # --------------------
            ts = agg.set_index("Year")["Rape"]
            if len(ts) >= 3:
                pred_df, model = forecast_series(ts, order=(1, 1, 1), steps=5)
                st.line_chart(pd.concat([ts, pred_df["mean"]]))

                future_avg = pred_df["mean"].mean()
                st.success(f"🔮 Predicted future rape cases per year ≈ **{future_avg:.1f}**")

            else:
                st.info("Not enough data for prediction.")


            # --------------------
            # SIMILAR AREAS
            # --------------------
            matrix = build_feature_matrix(df)
            try:
                recs = recommend_similar((state, district_sel if district_sel else ""), matrix)

                st.subheader("Similar Areas Based on Crime Pattern")
                for r, s in recs:
                    st.write(f"• {r} (Similarity score: {round(s,3)})")

                st.info("These areas show similar crime behavior. Safety strategies used there may also work here.")

            except:
                st.info("Showing high-risk districts in the same state instead.")
                top = (
                    df[df['STATE/UT'] == state]
                    .groupby("DISTRICT")[["Rape"]]
                    .sum()
                    .sort_values("Rape", ascending=False)
                    .head(5)
                )
                st.table(top)
