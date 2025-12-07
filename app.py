import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Credit Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    "<h1 style='color:white;'>📊 Credit Risk Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)


# ==========================
# LOAD LOCAL CSV FILE
# ==========================

@st.cache_data
def load_data():
    # IMPORTANT: the CSV MUST be inside the GitHub repo!
    df = pd.read_csv("application_data_sample.csv")

    # ---------------- Cleaning ----------------
    if "DAYS_EMPLOYED" in df.columns:
        df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    # Fill categorical
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")

    # Fill numeric
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ["SK_ID_CURR", "TARGET"]:
            df[col] = df[col].fillna(df[col].median())

    # ---------------- Feature Engineering ----------------
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).round(1)
    df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"] / 365).round(1)

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].mean(axis=1)
    df["SHORT_EMPLOYMENT"] = (df["YEARS_EMPLOYED"] < 1).astype(int)

    return df


df = load_data()


# ==========================
# SIDEBAR FILTERS
# ==========================

st.sidebar.header("🔍 Filters")

target_filter = st.sidebar.selectbox(
    "Filter by TARGET:",
    ["All", "Default only (1)", "Non-default only (0)"]
)

gender = st.sidebar.selectbox(
    "Gender:", ["All"] + sorted(df["CODE_GENDER"].unique())
)

income = st.sidebar.selectbox(
    "Income Type:", ["All"] + sorted(df["NAME_INCOME_TYPE"].unique())
)

age_min = float(df["AGE_YEARS"].min())
age_max = float(df["AGE_YEARS"].max())

age_range = st.sidebar.slider(
    "Age Range:",
    age_min, age_max,
    (age_min, age_max)
)


# ==========================
# APPLY FILTERS
# ==========================

df_filtered = df.copy()

if target_filter == "Default only (1)":
    df_filtered = df_filtered[df_filtered["TARGET"] == 1]
elif target_filter == "Non-default only (0)":
    df_filtered = df_filtered[df_filtered["TARGET"] == 0]

if gender != "All":
    df_filtered = df_filtered[df_filtered["CODE_GENDER"] == gender]

if income != "All":
    df_filtered = df_filtered[df_filtered["NAME_INCOME_TYPE"] == income]

df_filtered = df_filtered[
    (df_filtered["AGE_YEARS"] >= age_range[0]) &
    (df_filtered["AGE_YEARS"] <= age_range[1])
]


# ==========================
# KPI METRICS
# ==========================

st.subheader("📌 Key Metrics")

total_clients = len(df_filtered)
default_rate = df_filtered["TARGET"].mean() * 100 if total_clients > 0 else 0
avg_income = df_filtered["AMT_INCOME_TOTAL"].mean()
avg_credit = df_filtered["AMT_CREDIT"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Clients", f"{total_clients:,}")
c2.metric("Default Rate", f"{default_rate:.2f}%")
c3.metric("Average Income", f"{avg_income:,.0f}")
c4.metric("Average Credit", f"{avg_credit:,.0f}")

st.markdown("<hr>", unsafe_allow_html=True)


# ==========================
# CATEGORY DEFAULT RATE
# ==========================

st.subheader("📉 Default Rate by Category")

cat_col = st.selectbox(
    "Choose Category:",
    ["CODE_GENDER", "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE",
     "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"]
)

cat_group = (
    df_filtered.groupby(cat_col)["TARGET"]
    .mean()
    .reset_index()
    .rename(columns={"TARGET": "default_rate"})
)

fig1 = px.bar(
    cat_group,
    x=cat_col,
    y="default_rate",
    title=f"Default Rate by {cat_col}",
    template="plotly_dark"
)

st.plotly_chart(fig1, use_container_width=True)


# ==========================
# NUMERIC DISTRIBUTIONS
# ==========================

st.subheader("📈 Numeric Feature Distribution")

num_col = st.selectbox(
    "Choose Numeric Feature:",
    ["AGE_YEARS", "YEARS_EMPLOYED", "CREDIT_INCOME_RATIO",
     "ANNUITY_INCOME_RATIO", "EXT_SOURCE_MEAN",
     "AMT_INCOME_TOTAL", "AMT_CREDIT"]
)

fig2 = px.histogram(
    df_filtered,
    x=num_col,
    color="TARGET",
    nbins=40,
    template="plotly_dark",
    opacity=0.6,
    marginal="box"
)

st.plotly_chart(fig2, use_container_width=True)


# ==========================
# RAW DATA VIEWER
# ==========================

with st.expander("📄 Show Raw Data"):
    st.dataframe(df_filtered.head(200))
