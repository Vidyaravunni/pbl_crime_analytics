import streamlit as st
import pandas as pd
import numpy as np
import os
import time

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from io import BytesIO
except ImportError:
    st.error("Please install reportlab: pip install reportlab")
    st.stop()

from preprocess import load_and_clean, aggregate_by_area
from eda import plot_time_series, plot_top_crimes, plot_pie_composition, correlation_heatmap
from stats_utils import bootstrap_ci, two_sample_ttest
from ts_forecast import forecast_series
from similarity import build_feature_matrix, recommend_similar

DATA_PATH = "data/crime_data.csv"

# ==============================
# PDF REPORT GENERATION FUNCTION
# ==============================
def generate_pdf_report(agg, state, district):
    st.markdown("### 📄 Summary Report")
    
    # Create PDF buffer
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=30,
        alignment=1  # Center
    )
    
    story = []
    
    # Title
    story.append(Paragraph("Crime Analysis Summary", title_style))
    story.append(Spacer(1, 12))
    
    # Selected Region
    region_text = f"<b>State/UT:</b> {state}<br/>"
    if district:
        region_text += f"<b>District:</b> {district}"
    else:
        region_text += "<b>District:</b> All Districts"
    story.append(Paragraph(region_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Identify crime columns
    crime_cols = [c for c in agg.columns if c not in ["Year"]]
    
    # Peak Crime Year
    if crime_cols:
        agg["total_crimes"] = agg[crime_cols].sum(axis=1)
        peak_idx = agg["total_crimes"].idxmax()
        peak_year = int(agg.loc[peak_idx, "Year"])
        peak_value = int(agg.loc[peak_idx, "total_crimes"])
        
        peak_text = f"<b>Peak Crime Year:</b> {peak_year} — <b>Total Crimes:</b> {peak_value:,}"
        story.append(Paragraph(peak_text, styles['Heading2']))
        story.append(Spacer(1, 12))
        
        st.write(f"*Peak Crime Year:* {peak_year} — Total Crimes: {peak_value}")
    
    # Top Crimes Table
    total_by_type = agg[crime_cols].sum().sort_values(ascending=False).head(5)
    table_data = [['Crime Type', 'Total']]
    for crime, value in total_by_type.items():
        table_data.append([crime, f"{int(value):,}"])
    
    top_crimes_table = Table(table_data)
    top_crimes_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(Paragraph("Top 5 Crime Types", styles['Heading2']))
    story.append(top_crimes_table)
    story.append(Spacer(1, 12))
    
    st.write("*Top 5 Crimes:*")
    st.dataframe(total_by_type)
    
    # Build PDF
    doc.build(story)
    
    # Create download button for PDF
    pdf_buffer = buffer.getvalue()
    buffer.close()
    
    st.download_button(
        label="⬇ Download PDF Report",
        data=pdf_buffer,
        file_name="crime_summary_report.pdf",
        mime="application/pdf"
    )

# ===========================
# LOAD DATASET
# ===========================
@st.cache_data
def load_and_clean_data():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "crime_data.csv")
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    if 'Year' in df.columns:
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

    df = df.drop_duplicates()
    return df

df = load_and_clean_data()

# ===========================
# SESSION STATE
# ===========================
if "show_form" not in st.session_state:
    st.session_state.show_form = False

# ===========================
# SIDEBAR
# ===========================
st.sidebar.header("Add New Record")

# Button styling
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

button_label = "📝 Report a Crime" if not st.session_state.show_form else "🔙 Back to Dashboard"

if st.sidebar.button(button_label):
    st.session_state.show_form = not st.session_state.show_form
    st.rerun()

# ===========================
# MAIN PAGE
# ===========================
if not st.session_state.show_form:
    st.title("Crime Pattern Analysis — State/District Dashboard")

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
                st.warning("No data available for this selection.")
            else:
                st.subheader(f"Analysis for {state}" + (f" — {district_sel}" if district_sel else ""))

                st.plotly_chart(plot_time_series(agg, title="Crime counts over years"), use_container_width=True)
                st.plotly_chart(plot_top_crimes(agg), use_container_width=True)
                st.plotly_chart(plot_pie_composition(agg), use_container_width=True)
                st.plotly_chart(correlation_heatmap(agg), use_container_width=True)

                rape_series = agg['Rape'].values
                mean_rape, ci_rape = bootstrap_ci(rape_series, np.mean, n_boot=2000)
                st.write(f"*Average rape cases per year:* {float(mean_rape):.2f}")
                lower = float(ci_rape[0])
                upper = float(ci_rape[1])

                st.write(f"*95% Confidence Interval:* {lower:.2f} to {upper:.2f} cases per year")


                ts = agg.set_index('Year')['Rape']
                if len(ts) >= 3:
                    pred_df, model = forecast_series(ts, order=(1, 1, 1), steps=5)
                    fig_df = pd.concat([ts.rename('observed'), pred_df['mean'].rename('forecast')], axis=0)
                    st.line_chart(fig_df)
                else:
                    st.info("Not enough years to forecast (need 3+).")

                # Similarity recommendations
                matrix = build_feature_matrix(df)
                try:
                    recs = recommend_similar((state, district_sel if district_sel else ""), matrix)
                    st.subheader("Similar Areas (by crime profile)")
                    for r, s in recs:
                        st.write(r, round(s, 3))
                except Exception:
                    st.info("Could not compute similarity — showing top districts by Rape.")
                    top = (
                        df[df['STATE/UT'] == state]
                        .groupby('DISTRICT')[['Rape']]
                        .sum()
                        .sort_values('Rape', ascending=False)
                        .head(5)
                    )
                    st.table(top)

                # 🔥 *Generate PDF Summary Report at the END*
                generate_pdf_report(agg, state, district_sel)

else:
    st.subheader("Report a New Crime Record")

    with st.form("crime_report_form", clear_on_submit=True):
        st.write("### Enter Crime Details")

        state_ut = st.text_input("State/UT")
        district = st.text_input("District")
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

            csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "crime_data.csv")

            existing_df = pd.read_csv(csv_path)
            missing_cols = [c for c in existing_df.columns if c not in new_row]
            for col in missing_cols:
                new_row[col] = 0

            updated_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
            updated_df.to_csv(csv_path, index=False)

            st.success("✅ Record added successfully!")
            time.sleep(2)
            st.session_state.show_form = False
            st.rerun()


