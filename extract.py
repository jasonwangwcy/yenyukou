import pandas as pd

# ========== 安全讀檔 ==========
def safe_read_csv(path, skiprows=1):
    for enc in ["utf-8-sig", "utf-8", "big5", "cp950"]:
        try:
            df = pd.read_csv(path, encoding=enc, skiprows=skiprows)
            df.columns = df.columns.str.strip().str.replace("\ufeff", "")
            return df
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"無法讀取檔案：{path}")

# ========== 讀取並處理資料 ==========
# 讀入原始資料（跳過第 1 行）
df = safe_read_csv("Data/a/csvvv/0050.csv", skiprows=1)

# 選前 4 欄，並保留原始欄位名稱（含全形括號）
df = df.iloc[:, :4].copy()
df.columns = ["年月", "標的碼", "投資比率％", "投資金額（千元）"]  # 注意這裡是全形（）

# 過濾出 TT99 資料
df_tt99 = df[df["標的碼"] == "TT99"].copy()

# 整理欄位格式
df_tt99["年月"] = df_tt99["年月"].astype(str).str.strip()
df_tt99["投資金額（千元）"] = (
    df_tt99["投資金額（千元）"]
    .astype(str)
    .str.replace(",", "")
    .str.replace("-", "0")  # 如果有 dash，視為 0
    .astype(float)
)

# 篩選年月區間
df_tt99 = df_tt99[(df_tt99["年月"] >= "2016/02") & (df_tt99["年月"] <= "2025/02")]
df_tt99 = df_tt99.drop_duplicates(subset=["年月"])

# 最終結果欄位整理
df_result = df_tt99[["年月", "投資金額（千元）"]].reset_index(drop=True)

# 輸出 CSV
df_result.to_csv("TT99_FundSize_2016_2025.csv", index=False, encoding="utf-8-sig")
print("✅ 已輸出：TT99_FundSize_2016_2025.csv")
