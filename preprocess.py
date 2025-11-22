import pandas as pd
import pickle
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_FILE = "integrated_dataset_cleaned_final.parquet"
OUTPUT_STATIC_FILE = "static_results.pkl"

def process_data():
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_parquet(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}. Make sure it is in the same folder.")
        return

    # Ensure helper columns exist for calculations (Fast in memory)
    print("Preparing helper columns...")
    if "CRASH_DATETIME" in df.columns:
        # Ensure datetime format just in case, usually fast if already done
        if not pd.api.types.is_datetime64_any_dtype(df["CRASH_DATETIME"]):
             df["CRASH_DATETIME"] = pd.to_datetime(df["CRASH_DATETIME"], errors="coerce")
        
        df["YEAR"] = df["CRASH_DATETIME"].dt.year
        df["MONTH"] = df["CRASH_DATETIME"].dt.month_name()
        df["HOUR"] = df["CRASH_DATETIME"].dt.hour

    print("Calculating Static RQ1 - RQ10 data (This saves startup time)...")
    static_data = {}

    # --- RQ1: Street with Most Crashes per Borough ---
    if {"BOROUGH", "STREET NAME", "COLLISION_ID"}.issubset(df.columns):
        s1 = df[["BOROUGH", "STREET NAME", "COLLISION_ID"]].dropna()
        counts = s1.drop_duplicates(subset=["COLLISION_ID"]).groupby(["BOROUGH", "STREET NAME"], observed=True).size().reset_index(name="CrashCount")
        static_data["rq1"] = counts.sort_values("CrashCount", ascending=False).groupby("BOROUGH", observed=True).head(1).reset_index(drop=True)

    # --- RQ2: Day of Week ---
    if {"CRASH_DATETIME", "COLLISION_ID"}.issubset(df.columns):
        s2 = df[["CRASH_DATETIME", "COLLISION_ID"]].copy()
        s2["DAY_OF_WEEK"] = s2["CRASH_DATETIME"].dt.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        static_data["rq2"] = s2.drop_duplicates(subset=["COLLISION_ID"]).groupby("DAY_OF_WEEK", observed=True).size().reindex(day_order).fillna(0).reset_index(name="CrashCount")

    # --- RQ3: Bodily Injury (Killed) ---
    if {"BODILY_INJURY", "PERSON_INJURY"}.issubset(df.columns):
        s3 = df.loc[df["PERSON_INJURY"].astype(str).str.upper() == "KILLED", ["BODILY_INJURY"]].dropna()
        rq3 = s3.groupby(["BODILY_INJURY"], observed=True).size().reset_index(name="Count")
        rq3["Status"] = "Killed"
        static_data["rq3"] = rq3

    # --- RQ4: Position in Vehicle (Killed) ---
    if {"POSITION_IN_VEHICLE", "PERSON_INJURY"}.issubset(df.columns):
        s4 = df.loc[df["PERSON_INJURY"].astype(str).str.upper() == "KILLED", ["POSITION_IN_VEHICLE"]].dropna()
        ex = ["UNKNOWN", "DOES NOT APPLY", "NOT APPLICABLE", "N/A", "OTHER"]
        s4 = s4[~s4["POSITION_IN_VEHICLE"].astype(str).str.upper().isin(ex)]
        rq4 = s4.groupby(["POSITION_IN_VEHICLE"], observed=True).size().reset_index(name="Count")
        rq4["Status"] = "Killed"
        static_data["rq4"] = rq4

    # --- RQ5: Driver Age vs Time ---
    if {"PERSON_TYPE", "PERSON_AGE", "CRASH_DATETIME"}.issubset(df.columns):
        mask_driver = df["PERSON_TYPE"].astype(str).str.upper().str.contains("DRIVER", na=False)
        s5 = df.loc[mask_driver, ["PERSON_AGE", "CRASH_DATETIME"]].copy()
        s5["HOUR"] = s5["CRASH_DATETIME"].dt.hour
        s5["TIME_OF_DAY"] = s5["HOUR"].apply(lambda h: "Night" if (pd.notna(h) and (h >= 20 or h < 6)) else "Day")
        
        def age_bucket(a):
            try:
                a = float(a)
                if a < 18: return "<18"
                elif a <= 25: return "18–25"
                elif a <= 40: return "26–40"
                elif a <= 60: return "41–60"
                else: return "60+"
            except: return "Unknown"

        s5["AgeGroup"] = s5["PERSON_AGE"].apply(age_bucket)
        static_data["rq5"] = s5.groupby(["AgeGroup", "TIME_OF_DAY"], observed=True).size().reset_index(name="Count")

    # --- RQ6: Pedestrian Complaints ---
    if {"PERSON_TYPE", "PERSON_INJURY", "COMPLAINT"}.issubset(df.columns):
        mask_ped = df["PERSON_TYPE"].astype(str).str.upper() == "PEDESTRIAN"
        s6 = df.loc[mask_ped, ["PERSON_INJURY", "COMPLAINT"]]
        s6 = s6[~s6["PERSON_INJURY"].astype(str).str.upper().isin(["NO APPARENT INJURY", "UNKNOWN"])].dropna(subset=["COMPLAINT"])
        ex_comp = ["DOES NOT APPLY", "UNKNOWN", "NOT APPLICABLE", "N/A"]
        s6 = s6[~s6["COMPLAINT"].astype(str).str.upper().isin(ex_comp)]
        static_data["rq6"] = s6.groupby("COMPLAINT", observed=True).size().reset_index(name="Count").sort_values("Count", ascending=True).tail(10)

    # --- RQ7: Contributing Factors by Sex ---
    f_col = "CONTRIBUTING FACTOR VEHICLE 1"
    if {"PERSON_SEX", f_col}.issubset(df.columns):
        mask_sex = df["PERSON_SEX"].astype(str).str.upper().isin(["F", "M", "FEMALE", "MALE"])
        s7 = df.loc[mask_sex, ["PERSON_SEX", f_col]]
        s7 = s7[~s7[f_col].astype(str).str.upper().str.contains("UNSPECIFIED|UNKNOWN", na=False)]
        rq7 = s7.groupby(["PERSON_SEX", f_col], observed=True).size().reset_index(name="Count")
        static_data["rq7"] = rq7.sort_values(["PERSON_SEX", "Count"], ascending=[True, False]).groupby("PERSON_SEX").head(5)

    # --- RQ8: Safety Equipment ---
    if {"SAFETY_EQUIPMENT", "PERSON_TYPE"}.issubset(df.columns):
        mask_d_sq = df["PERSON_TYPE"].astype(str).str.upper().str.contains("DRIVER", na=False)
        s8 = df.loc[mask_d_sq, ["SAFETY_EQUIPMENT"]].dropna()
        static_data["rq8"] = s8.groupby("SAFETY_EQUIPMENT", observed=True).size().reset_index(name="Count")

    # --- RQ9: Pedestrian Bodily Injury ---
    if {"PERSON_TYPE", "BODILY_INJURY"}.issubset(df.columns):
        mask_ped_bi = df["PERSON_TYPE"].astype(str).str.upper() == "PEDESTRIAN"
        s9 = df.loc[mask_ped_bi, ["BODILY_INJURY"]].dropna()
        ex_inj = ["UNKNOWN", "DOES NOT APPLY", "NOT APPLICABLE", "N/A", "UNSPECIFIED"]
        s9 = s9[~s9["BODILY_INJURY"].astype(str).str.upper().isin(ex_inj)]
        static_data["rq9"] = s9.groupby("BODILY_INJURY", observed=True).size().reset_index(name="Count").sort_values("Count", ascending=False).head(10)

    # --- RQ10: Crash Hour ---
    if {"CRASH_DATETIME", "COLLISION_ID"}.issubset(df.columns):
        s10 = df[["CRASH_DATETIME", "COLLISION_ID"]].drop_duplicates(subset=["COLLISION_ID"]).copy()
        s10["HOUR"] = s10["CRASH_DATETIME"].dt.hour
        static_data["rq10"] = s10.groupby("HOUR", observed=True).size().reset_index(name="CrashCount")

    print(f"Saving static results to {OUTPUT_STATIC_FILE}...")
    with open(OUTPUT_STATIC_FILE, "wb") as f:
        pickle.dump(static_data, f)

    print("Done! Upload 'static_results.pkl' to your Space (alongside your existing parquet file).")

if __name__ == "__main__":
    process_data()