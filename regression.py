import pandas as pd
import numpy as np
import statsmodels.api as sm

# ========== 函式區 ==========
def safe_read_csv(path, skiprows=None):
    """
    嘗試多種編碼讀取 CSV 並跳過指定的列，並清理欄位名稱
    """
    for enc in ["utf-8-sig", "utf-8", "big5", "cp950"]:
        try:
            df = pd.read_csv(path, encoding=enc, skiprows=skiprows)
            # 移除 BOM 和前後空白
            df.columns = df.columns.str.strip().str.replace("\ufeff", "")
            return df
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"無法讀取檔案：{path}")

def gini_coefficient(w):
    """以 Lorenz 曲線方式計算 Gini 係數"""
    sorted_w = np.sort(w)
    n = len(w)
    cum_w = np.cumsum(sorted_w)
    gini = (n + 1 - 2 * np.sum(cum_w) / cum_w[-1]) / n
    return gini

# ========== Step 1. 處理價格資料（僅抓 "收盤價(元)"） ==========
# 讀取價格資料檔：跳過第一行（檔案標題），以第二行作為 header
df_price = safe_read_csv("Data/b/csvb/0050b.csv", skiprows=1)
# 根據檔案結構，僅選取 "年月" 與 "收盤價(元)" 欄位
df_price = df_price[["年月", "收盤價(元)"]].dropna()
# 清理數值：將 "收盤價(元)" 移除逗號並轉為 float；清理 "年月"
df_price["收盤價(元)"] = df_price["收盤價(元)"].astype(str).str.replace(",", "").astype(float)
df_price["年月"] = df_price["年月"].astype(str).str.strip()
# 排序依據 "年月" 並計算 log return： log(P_next / P_current)
df_price = df_price.sort_values("年月")
df_price["log_return"] = np.log(df_price["收盤價(元)"].shift(-1) / df_price["收盤價(元)"])
df_price = df_price.dropna()

# ========== Step 2. 處理持股資料（計算集中度與 FundSize） ==========
df_holdings = safe_read_csv("Data/a/csvvv/0050 .csv", skiprows=1)
# 假設前4欄依序為： "年月", "標的碼", "投資比率％", "投資金額(千元)" ，只讀取這幾欄：
df_holdings = df_holdings.iloc[:, :4].copy()
df_holdings.columns = ["年月", "標的碼", "投資比率％", "投資金額(千元)"]
df_holdings = df_holdings.dropna(subset=["年月", "標的碼", "投資比率％", "投資金額(千元)"])
df_holdings["投資比率％"] = df_holdings["投資比率％"].astype(str).str.replace(",", "").astype(float)
df_holdings["投資金額(千元)"] = df_holdings["投資金額(千元)"].astype(str).str.replace(",", "").astype(float)
df_holdings["年月"] = df_holdings["年月"].astype(str).str.strip()

# 只保留標的碼為純數字的（代表個股）或 "TT99"（代表基金規模）
df_holdings = df_holdings[
    df_holdings["標的碼"].astype(str).str.isnumeric() | (df_holdings["標的碼"] == "TT99")
]

# 依據每月資料計算集中度指標與 FundSize
records = []
months = sorted(df_holdings["年月"].unique())
for month in months:
    month_df = df_holdings[df_holdings["年月"] == month]
    
    # 集中度計算：排除 TT99，取前 10 大持股
    filtered = month_df[month_df["標的碼"] != "TT99"]
    filtered = filtered.sort_values(by="投資比率％", ascending=False)
    filtered = filtered.drop_duplicates(subset=["標的碼"]).iloc[:10]
    if len(filtered) < 10:
        continue
    w = filtered["投資比率％"].values / 100
    hhi = np.sum(w**2)
    gini = gini_coefficient(w)
    entropy = -np.sum(w * np.log(w))
    
    # FundSize 為該月中標的碼為 "TT99" 的投資金額(千元)
    fund_row = month_df[month_df["標的碼"] == "TT99"]
    fund_size = fund_row["投資金額(千元)"].values[0] if not fund_row.empty else np.nan
    
    records.append({
        "年月": month,
        "HHI": hhi,
        "Gini": gini,
        "Entropy": entropy,
        "FundSize": fund_size
    })

df_factors = pd.DataFrame(records).dropna()

# ========== Step 3. 合併價格與持股指標 ==========
df_merged = pd.merge(df_price, df_factors, on="年月", how="inner")

# ========== Step 4. 單變數回歸模型 ==========
# 以 log_return 為依變數；自變數分別用集中度指標（HHI, Gini, Entropy），並控制 FundSize
regression_results = {}
for factor in ["HHI", "Gini", "Entropy"]:
    X = df_merged[[factor, "FundSize"]]
    X = sm.add_constant(X)
    y = df_merged["log_return"]
    model = sm.OLS(y, X).fit()
    regression_results[f"log_return ~ {factor} + FundSize"] = model

# ========== Step 5. 匯總回歸結果 ==========
summary = {
    name: {
        "R²": model.rsquared,
        "coef": model.params.to_dict(),
        "p-value": model.pvalues.to_dict()
    }
    for name, model in regression_results.items()
}
summary_df = pd.DataFrame(summary).T
print(summary_df)

# 如果需要儲存結果，可取消下行註解
# summary_df.to_csv("result/0050_regression_summary.csv", index=True)
