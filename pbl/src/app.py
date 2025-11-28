import streamlit as st
import pandas as pd
import numpy as np
import os
import time

from preprocess import load_and_clean, aggregate_by_area
from eda import plot_time_series, plot_top_crimes, plot_pie_composition, correlation_heatmap
from stats_utils import bootstrap_ci, two_sample_ttest
from ts_forecast import forecast_series
from similarity import build_feature_matrix, recommend_similar

DATA_PATH = "data/crime_data.csv"


@st.cache_data
def load_and_clean(path=DATA_PATH):
    path = os.path.join(os.path.dirname(_file_), "..", "data", "crime_data.csv")
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Convert Year to int (if needed)
    if 'Year' in df.columns:
        df['Year'] = df['Year'].astype(int, errors='ignore')

    # Remove rows where District == 'Total'
    df = df[df['DISTRICT'].str.strip().str.lower() != 'total']

    # Normalize STATE/UT names
    df['STATE/UT'] = df['STATE/UT'].str.strip().str.title()

    # Fix known duplicates / inconsistent names
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

    # Drop duplicates
    df = df.drop_duplicates()

    return df


# Load dataset
df = load_and_clean()

# ===== Initialize session state =====
if "show_form" not in st.session_state:
    st.session_state.show_form = False

# ===== Sidebar section =====
st.sidebar.header("Add New Record")

# ---- Custom CSS for Red Button ----
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #e63946;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
    }
    div.stButton > button:first-child:hover {
        background-color: #d62828;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Dynamic button name
button_label = "📝 Report a Crime" if not st.session_state.show_form else "🔙 Back to Dashboard"

if st.sidebar.button(button_label):
    st.session_state.show_form = not st.session_state.show_form
    st.rerun()

# ===== MAIN CONTENT =====

if not st.session_state.show_form:
    # -------------------- Dashboard --------------------
    st.title("Crime Pattern Analysis — State/District Dashboard")

    # Input selectors
    states = sorted(df['STATE/UT'].unique())
    state = st.selectbox("Select State/UT", [""] + states)

    districts = []
    if state:
        districts = sorted(df[df['STATE/UT'] == state]['DISTRICT'].unique())
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

                # Bootstrap CI example for Rape
                rape_series = agg['Rape'].values
                mean_rape, ci_rape = bootstrap_ci(rape_series, np.mean, n_boot=2000)
                st.write("Rape — mean per year:", round(mean_rape, 2))
                st.write("95% bootstrap CI:", (round(ci_rape[0], 2), round(ci_rape[1], 2)))

                # Forecast example (Rape)
                ts = agg.set_index('Year')['Rape']
                if len(ts) >= 3:
                    pred_df, model = forecast_series(ts, order=(1, 1, 1), steps=5)
                    fig_df = pd.concat([ts.rename('observed'), pred_df['mean'].rename('forecast')], axis=0)
                    st.line_chart(fig_df)
                else:
                    st.info("Not enough years to forecast reliably (need >=3).")

                # Similarity recommendation
                matrix = build_feature_matrix(df)
                try:
                    recs = recommend_similar((state, district_sel if district_sel else ""), matrix)
                    st.subheader("Similar areas (by crime profile)")
                    for r, s in recs:
                        st.write(r, round(s, 3))
                except Exception:
                    st.info("Could not compute similarity for this selection; showing top districts in same state.")
                    top = (
                        df[df['STATE/UT'] == state]
                        .groupby('DISTRICT')[['Rape']]
                        .sum()
                        .sort_values('Rape', ascending=False)
                        .head(5)
                    )
                    st.table(top)

    # Sidebar information
    st.sidebar.header("Project Modules")
    st.sidebar.markdown("""
    - EDA: time series, heatmaps  
    - Stats: bootstrap CI, t-tests  
    - Forecasting: ARIMA  
    - Similarity & recommender  
    - Optional: NLP, network analysis  
    """)

else:
    # -------------------- Report Form --------------------
    st.subheader("Report a New Crime Record")

    with st.form("crime_report_form", clear_on_submit=True):
        st.write("### Enter Crime Details")

        # Dropdowns and inputs
        state_ut = st.selectbox("Select State", sorted(df['STATE/UT'].unique()))
        district = st.text_input("District Name")
        year = st.number_input("Year", min_value=2001, max_value=2030, value=2025)

        st.markdown("#### Enter Crime Counts:")
        rape = st.number_input("Rape", min_value=0, value=0)
        kidnap = st.number_input("Kidnapping and Abduction", min_value=0, value=0)
        dowry = st.number_input("Dowry Deaths", min_value=0, value=0)
        assault = st.number_input("Assault on women with intent to outrage her modesty", min_value=0, value=0)
        insult = st.number_input("Insult to modesty of Women", min_value=0, value=0)
        cruelty = st.number_input("Cruelty by Husband or his Relatives", min_value=0, value=0)

        submitted = st.form_submit_button("Submit Record")

        if submitted:
            new_row = {
                'STATE/UT': state_ut.strip().title(),
                'DISTRICT': district.strip().title(),
                'Year': int(year),
                'Rape': int(rape),
                'Kidnapping and Abduction': int(kidnap),
                'Dowry Deaths': int(dowry),
                'Assault on women with intent to outrage her modesty': int(assault),
                'Insult to modesty of Women': int(insult),
                'Cruelty by Husband or his Relatives': int(cruelty)
            }

            csv_path = os.path.join(os.path.dirname(_file_), "..", "data", "crime_data.csv")

            # Append directly to CSV
            existing_df = pd.read_csv(csv_path)
            missing_cols = [c for c in existing_df.columns if c not in new_row]
            for col in missing_cols:
                new_row[col] = 0

            updated_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
            updated_df.to_csv(csv_path, index=False)

            st.success("✅ Report added successfully!")
            time.sleep(2)
            st.session_state.show_form = False
            st.rerun()
