import streamlit as st
import numpy as np
import pandas as pd

st.title("The Lexical Observatory")

slots = ["noun_or", "verb_fy", "adj_id", "adj_ic", "adj_ible"]

families = {
    "horror": ["horror", "horrify", "horrid", "horrific", "horrible"],
    "stupor": ["stupor", "stupefy", "stupid", None,       None],
    "tepor":  ["tepor",  "tepefy",  "tepid",  None,       None],
    "terror": ["terror", "terrify", None,     "terrific", "terrible"]
}

st.subheader("1. Controls: Expand or Refine the Dataset")
include_candor = st.checkbox("Add 'candor' family (candor, candid — no -fy verb)")
include_sopor = st.checkbox("Add 'sopor' family (sopor, soporify, soporific — no -ible adj)")
use_frequency = st.checkbox("Use 0/1/2 Frequency Scale (1 = rare 'tepor/tepefy/sopor/soporify', 2 = common)")

if include_candor:
    families["candor"] = ["candor", None, "candid", None, None]
if include_sopor:
    families["sopor"] = ["sopor", "soporify", None, "soporific", None]

df_words = pd.DataFrame.from_dict(families, orient="index", columns=slots)

st.subheader("2. Lexical Grid (Word Forms)")
st.table(df_words)

# Build Binary (0/1) or Frequency (0/1/2) Matrix A
A = df_words.notna().astype(int).to_numpy()
if use_frequency:
    A = A * 2
    A[2, 0] = 1            # "tepor" is rare -> 1
    A[2, 1] = 1            # "tepefy" is rare -> 1
    if include_sopor:
        sopor_row = list(families.keys()).index("sopor")
        A[sopor_row, 0] = 1  # "sopor" is rare -> 1
        A[sopor_row, 1] = 1  # "soporify" is rare -> 1

df_matrix = pd.DataFrame(A, index=families.keys(), columns=slots)

st.subheader("3. Attestation / Frequency Matrix (A)")
st.table(df_matrix)

st.subheader("4. Marginal Sums")
st.write("Row sums (per family):", A.sum(axis=1))
st.write("Column sums (per slot):", A.sum(axis=0))

st.subheader("5. Matrix Geometry Explorer")
rank = np.linalg.matrix_rank(A)
num_rows, num_cols = A.shape

st.write(f"Matrix shape: {num_rows} rows × {num_cols} columns")
st.write(f"Rank (independent dimensions): {rank}")
st.write(f"Null space dimension (column redundancies = cols - rank): {num_cols - rank}")
st.write(f"Left null space dimension (row redundancies = rows - rank): {num_rows - rank}")

st.subheader("6. Similarity (A @ A.T) and Co-occurrence (A.T @ A)")

family_names = list(families.keys())

# Family-by-Family Similarity Matrix (rows dot-producted with rows)
sim_matrix = A @ A.T
df_sim = pd.DataFrame(sim_matrix, index=family_names, columns=family_names)
st.write("**Family Similarity Matrix (A @ A.T):**")
st.table(df_sim)

# Slot-by-Slot Co-occurrence Matrix (columns dot-producted with columns)
cooc_matrix = A.T @ A
df_cooc = pd.DataFrame(cooc_matrix, index=slots, columns=slots)
st.write("**Suffix Co-occurrence Matrix (A.T @ A):**")
st.table(df_cooc)

# Cosine Similarity Matrix (normalized by vector lengths)
row_lengths = np.sqrt(np.diag(sim_matrix)).reshape(-1, 1)
cosine_sim = sim_matrix / (row_lengths @ row_lengths.T)
df_cosine = pd.DataFrame(np.round(cosine_sim, 2), index=family_names, columns=family_names)
st.write("**Normalized Family Similarity (Cosine Similarity):**")
st.table(df_cosine)