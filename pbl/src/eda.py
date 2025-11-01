# src/eda.py
import plotly.express as px
import pandas as pd
import numpy as np

from preprocess import aggregate_by_area, CRIME_COLS

def plot_time_series(agg_df, title=None):
    # long format for plotly
    df = agg_df.melt(id_vars='Year', value_vars=CRIME_COLS, var_name='Crime', value_name='Count')
    fig = px.line(df, x='Year', y='Count', color='Crime', markers=True, title=title)
    return fig

def plot_top_crimes(agg_df):
    # sum over years to get totals, then bar
    totals = agg_df[CRIME_COLS].sum().sort_values(ascending=False).reset_index()
    totals.columns = ['Crime', 'Total']
    fig = px.bar(totals, x='Crime', y='Total', title='Total crimes (selected area)', text='Total')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_pie_composition(agg_df):
    totals = agg_df[CRIME_COLS].sum()
    fig = px.pie(values=totals.values, names=totals.index, title='Crime composition (selected area)')
    return fig

def correlation_heatmap(df):
    corr = df[CRIME_COLS].corr()
    fig = px.imshow(corr, text_auto=True, title='Correlation between crime types')
    return fig
