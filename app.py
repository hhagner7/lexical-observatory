import streamlit as st
import numpy as np
import pandas as pd

st.title("The Lexical Observatory")

slots = ["noun", "verb_fy", "adj_id", "adj_ic", "adj_ible"]

families = {
    "horror": ["horror", "horrify", "horrid", "horrific", "horrible"],
    "stupor": ["stupor", "stupefy", "stupid", None,       None],
    "tepor":  ["tepor",  "tepefy",  "tepid",  None,       None],
    "terror": ["terror", "terrify", None,     "terrific", "terrible"]
}

# Build a labeled table where rows = word families, columns = slots
df_words = pd.DataFrame.from_dict(families, orient="index", columns=slots)

st.subheader("1. Lexical Grid (Word Forms)")
st.table(df_words)

# Convert to a binary NumPy matrix: 1 if attested (not None), 0 if missing (None)
A = df_words.notna().astype(int).to_numpy()
df_matrix = pd.DataFrame(A, index=families.keys(), columns=slots)

st.subheader("2. Binary Attestation Matrix (A)")
st.table(df_matrix)

st.subheader("3. Marginal Sums")
st.write("Forms per family (row sums):", A.sum(axis=1))
st.write("Productivity per slot (column sums):", A.sum(axis=0))