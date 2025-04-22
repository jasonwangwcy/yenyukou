import pandas as pd
import numpy as np
import os
import glob

# ✅ Gini 計算函數
def gini_coefficient(w):
    sorted_w = np.sort(w)
    n = len(w)
    cum_w = np.cumsum(sorted_w)
    gini = (n + 1 - 2 * np.sum(cum_w) / cum_w[-1]) / n
    return gini

# ✅ 資料夾設定
csv_folder = "Data/a/csvvv/"
output_folder = "result/"
os.makedirs(output_folder, exist_ok=True)
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

for file_path in csv_files:
    try:
        df_csv = pd.read_csv(file_path)
        df_clean = df_csv.iloc[2:].copy()
        df_clean.columns = df_csv.iloc[1]
        df_clean = df_clean.reset_index(drop=True)

        # 資料清理
        df_clean = df_clean.dropna(subset=["投資比率％"])
        df_clean["投資比率％"] = df_clean["投資比率％"].astype(str).str.replace(",", "").astype(float)
        df_clean["年月"] = df_clean["年月"].astype(str)

        # ✅ 只保留「標的碼為純數字」的資料
        df_clean = df_clean[df_clean["標的碼"].astype(str).str.isnumeric()]

        # 分析月份
        target_months = sorted([m for m in df_clean["年月"].unique() if "2016/02" <= m <= "2025/02"])
        records = []

        for month in target_months:
            month_df = df_clean[df_clean["年月"] == month].copy()

            # 排序 + 去除重複標的碼，保留前10名
            month_df = month_df.sort_values(by="投資比率％", ascending=False)
            month_df = month_df.drop_duplicates(subset=["標的碼"]).iloc[:10]

            w = month_df["投資比率％"].values / 100
            if len(w) < 10:
                continue

            hhi = np.sum(w**2)
            gini = gini_coefficient(w)
            entropy = -np.sum(w * np.log(w))

            records.append({
                "年月": month,
                "HHI": hhi,
                "Gini": gini,
                "Entropy": entropy
            })

        result_df = pd.DataFrame(records)
        code = os.path.basename(file_path).replace(".csv", "")
        result_df.to_csv(os.path.join(output_folder, f"{code}_concentration.csv"), index=False)
        print(f"✅ {code} 完成（僅處理標的碼為數字）")

    except Exception as e:
        print(f"❌ {file_path} 發生錯誤：{e}")
