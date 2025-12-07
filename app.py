import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ==========================
# CONFIGURATION
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
# LOAD DATA FROM GOOGLE DRIVE
# ==========================
@st.cache_data
def load_data():
    FILE_ID = "1Yhd72wyvDBbNwIUIck7LAOMLWR7TZhg6"
    FILE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

    df = pd.read_csv(FILE_URL)

    # Fix anomalies
    df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    # Fill categorical
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")

    # Fill numeric
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ["SK_ID_CURR", "TARGET"]:
            df[col] = df[col].fillna(df[col].median())

    # Feature Engineering
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).round(1)
    df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"] / 365).round(1)

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["SHORT_EMPLOYMENT"] = (df["YEARS_EMPLOYED"] < 1).astype(int)

    return df

df = load_data()

# ==========================
# SIDEBAR FILTERS
# ==========================

st.sidebar.header("🔍 Filters")

target_filter = st.sidebar.selectbox(
    "TARGET Filter:",
    ["All", "Default only (1)", "Non-default only (0)"]
)

gender_filter = st.sidebar.selectbox(
    "Gender:",
    ["All"] + sorted(df["CODE_GENDER"].unique().tolist())
)

income_filter = st.sidebar.selectbox(
    "Income Type:",
    ["All"] + sorted(df["NAME_INCOME_TYPE"].unique().tolist())
)

age_min = float(df["AGE_YEARS"].min())
age_max = float(df["AGE_YEARS"].max())

age_range = st.sidebar.slider(
    "Age range:",
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

if gender_filter != "All":
    df_filtered = df_filtered[df_filtered["CODE_GENDER"] == gender_filter]

if income_filter != "All":
    df_filtered = df_filtered[df_filtered["NAME_INCOME_TYPE"] == income_filter]

df_filtered = df_filtered[
    (df_filtered["AGE_YEARS"] >= age_range[0]) &
    (df_filtered["AGE_YEARS"] <= age_range[1])
]

# ==========================
# KPIs
# ==========================

st.subheader("📌 Key Metrics")

total_clients = len(df_filtered)
default_rate = df_filtered["TARGET"].mean() * 100 if total_clients > 0 else 0
avg_income = df_filtered["AMT_INCOME_TOTAL"].mean() if total_clients > 0 else 0
avg_credit = df_filtered["AMT_CREDIT"].mean() if total_clients > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Clients", f"{total_clients:,}")
c2.metric("Default Rate", f"{default_rate:.2f}%")
c3.metric("Avg Income", f"{avg_income:,.0f}")
c4.metric("Avg Credit", f"{avg_credit:,.0f}")

st.markdown("<hr>", unsafe_allow_html=True)

# ==========================
# CATEGORY DEFAULT RATE
# ==========================

st.subheader("📉 Default Rate by Category")

cat_col = st.selectbox(
    "Choose category:",
    ["CODE_GENDER", "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE",
     "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"]
)

cat_group = (
    df_filtered.groupby(cat_col)["TARGET"]
    .mean()
    .reset_index()
    .rename(columns={"TARGET": "default_rate"})
)

fig_cat = px.bar(
    cat_group,
    x=cat_col,
    y="default_rate",
    title=f"{cat_col} — Default Rate",
    template="plotly_dark"
)

st.plotly_chart(fig_cat, use_container_width=True)

# ==========================
# NUMERIC DISTRIBUTION
# ==========================

st.subheader("📈 Numeric Feature Distribution")

num_col = st.selectbox(
    "Choose numeric feature:",
    ["AGE_YEARS", "YEARS_EMPLOYED", "CREDIT_INCOME_RATIO",
     "ANNUITY_INCOME_RATIO", "EXT_SOURCE_MEAN",
     "AMT_INCOME_TOTAL", "AMT_CREDIT"]
)

fig_num = px.histogram(
    df_filtered,
    x=num_col,
    color="TARGET",
    nbins=40,
    opacity=0.6,
    template="plotly_dark",
    marginal="box"
)

st.plotly_chart(fig_num, use_container_width=True)

# ==========================
# RAW DATA
# ==========================

with st.expander("📄 Show Raw Data"):
    st.dataframe(df_filtered.head(300))

