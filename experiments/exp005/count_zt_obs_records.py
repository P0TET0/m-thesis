from pathlib import Path
import pandas as pd

# DF法とML法を同じ条件で比較したファイル
path = Path("data/output/starrydata2_step22_fitting_vs_ml_comparison/step22_row_level_comparison.csv")

df = pd.read_csv(path, low_memory=False)

print("columns:")
print(df.columns.tolist())

# ----------------------------
# 列名を自動で探す関数
# ----------------------------
def find_col(df, include_terms, exclude_terms=None):
    if exclude_terms is None:
        exclude_terms = []
    for col in df.columns:
        c = col.lower()
        if all(term.lower() in c for term in include_terms) and not any(term.lower() in c for term in exclude_terms):
            return col
    return None

sample_col = find_col(df, ["sample_key"])
zt_obs_col = find_col(df, ["zt", "obs"], exclude_terms=["pred", "mape", "error", "calc", "classification"])
zt_df_pred_col = (
    find_col(df, ["zt", "pred", "fitting"], exclude_terms=["mape", "error"])
    or find_col(df, ["zt", "pred", "df"], exclude_terms=["mape", "error"])
    or find_col(df, ["zt", "pred", "direct"], exclude_terms=["mape", "error"])
)
zt_ml_pred_col = find_col(df, ["zt", "pred", "ml"], exclude_terms=["mape", "error"])

print("\nDetected columns")
print("sample_col:", sample_col)
print("zt_obs_col:", zt_obs_col)
print("zt_df_pred_col:", zt_df_pred_col)
print("zt_ml_pred_col:", zt_ml_pred_col)

if sample_col is None:
    raise ValueError("sample_key列が見つかりません。columnsを確認してください。")
if zt_obs_col is None:
    raise ValueError("ZT_obs列が見つかりません。columnsを確認してください。")

# ZT_obsを数値化
df[zt_obs_col] = pd.to_numeric(df[zt_obs_col], errors="coerce")

# ----------------------------
# 温度点ベースのカウント
# ----------------------------
zt_rows = df[df[zt_obs_col].notna()].copy()

row_total = len(zt_rows)
row_high = (zt_rows[zt_obs_col] >= 1).sum()

print("\n=== 温度点ベース ===")
print("ZT_obsがある行数:", row_total)
print("ZT_obs >= 1 の行数:", row_high)
print("ZT_obs >= 1 の割合:", row_high / row_total if row_total else None)

# ----------------------------
# 試料ベースのカウント
# 各試料の最大ZTで判定
# ----------------------------
sample_zt = (
    zt_rows
    .groupby(sample_col, as_index=False)[zt_obs_col]
    .max()
    .rename(columns={zt_obs_col: "zt_obs_max"})
)

sample_total = len(sample_zt)
sample_high = (sample_zt["zt_obs_max"] >= 1).sum()

print("\n=== 試料ベース ===")
print("ZT_obsがある試料数:", sample_total)
print("最大ZT_obs >= 1 の試料数:", sample_high)
print("最大ZT_obs >= 1 の割合:", sample_high / sample_total if sample_total else None)

# ----------------------------
# DF法・ML法で実際に比較可能だった行数
# ----------------------------
def summarize_method(name, pred_col):
    if pred_col is None:
        print(f"\n{name}: 予測ZT列が見つかりませんでした")
        return
    
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    sub = df[df[zt_obs_col].notna() & df[pred_col].notna()].copy()
    
    row_total = len(sub)
    row_high = (sub[zt_obs_col] >= 1).sum()
    
    sample_zt = (
        sub
        .groupby(sample_col, as_index=False)[zt_obs_col]
        .max()
        .rename(columns={zt_obs_col: "zt_obs_max"})
    )
    sample_total = len(sample_zt)
    sample_high = (sample_zt["zt_obs_max"] >= 1).sum()
    
    print(f"\n=== {name}で評価可能だったZT_obs ===")
    print("温度点ベース ZT_obs行数:", row_total)
    print("温度点ベース ZT_obs>=1 行数:", row_high)
    print("試料ベース ZT_obsあり試料数:", sample_total)
    print("試料ベース 最大ZT_obs>=1 試料数:", sample_high)

summarize_method("DF法", zt_df_pred_col)
summarize_method("ML法", zt_ml_pred_col)