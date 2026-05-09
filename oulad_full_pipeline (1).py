"""
OULAD Full Data Preprocessing Pipeline
=======================================
Covers all 7 CSV files:
    1. studentInfo.csv
    2. studentRegistration.csv
    3. studentAssessment.csv
    4. assessments.csv
    5. studentVle.csv
    6. vle.csv
    7. courses.csv

Pipeline steps:
    - Missing value handling
    - Encoding
    - Outlier detection & capping
    - Aggregation
    - Merging
    - Scaling
    - Feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

hidden_nulls = ['N/A', 'n/a', 'NA', 'na', 'unknown', 'Unknown',
                '?', '', ' ', 'none', 'None', 'NULL', 'null']


def cap_outliers_iqr(df, col, lower=True, upper=True):
    """
    Cap outliers using the IQR method.
    Values below Q1-1.5*IQR or above Q3+1.5*IQR are capped (not dropped).
    Capping preserves rows while removing extreme influence.
    If IQR is 0 (column has no spread), skip capping to avoid destroying data.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # If IQR is 0, the column is heavily concentrated at one value
    # Capping would destroy all variation — skip it
    if IQR == 0:
        print(f"\n[Outlier capping] '{col}' skipped — IQR is 0, no spread to cap")
        return df

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    before = df[col].describe()

    if lower:
        df[col] = df[col].clip(lower=lower_bound)
    if upper:
        df[col] = df[col].clip(upper=upper_bound)

    after = df[col].describe()
    print(f"\n[Outlier capping] '{col}'")
    print(f"  Bounds → lower: {lower_bound:.2f}, upper: {upper_bound:.2f}")
    print(f"  Before max: {before['max']:.2f} | After max: {after['max']:.2f}")
    return df


# ═══════════════════════════════════════════════
# 1. studentInfo.csv
# ═══════════════════════════════════════════════
def process_student_info():
    df = pd.read_csv(r'C:\Users\kordo\Downloads\archive\studentInfo.csv', na_values=hidden_nulls)
    print(f"\n{'='*50}\nLoaded: studentInfo.csv → shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")

    # ── Missing values ──────────────────────────
    # imd_band: MAR — missing pattern tied to region
    # Fill with mode within each region group
    df['imd_band'] = df.groupby('region')['imd_band'].transform(
        lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x
    )

    # ── imd_band: convert ranges to midpoints ───
    # '10-20%' → 15.0  (makes it numerical and ordinal)
    def range_to_midpoint(val):
        if pd.isnull(val):
            return np.nan
        val = str(val).replace('%', '')
        parts = val.split('-')
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
        return np.nan

    df['imd_band'] = df['imd_band'].apply(range_to_midpoint)

    # ── Encoding: ordinal columns ────────────────
    # age_band: natural order → encode as integers
    age_map = {'0-35': 0, '35-55': 1, '55<=': 2}
    df['age_band'] = df['age_band'].map(age_map)

    # highest_education: natural order
    edu_map = {
        'No Formal quals': 0,
        'Lower Than A Level': 1,
        'A Level or Equivalent': 2,
        'HE Qualification': 3,
        'Post Graduate Qualification': 4
    }
    df['highest_education'] = df['highest_education'].map(edu_map)

    # final_result: meaningful scale
    # Withdrawn=-1 (left early), Fail=0, Pass=1, Distinction=2
    result_map = {'Withdrawn': -1, 'Fail': 0, 'Pass': 1, 'Distinction': 2}
    df['final_result'] = df['final_result'].map(result_map)

    # ── Encoding: binary columns ─────────────────
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})
    df['disability'] = df['disability'].map({'N': 0, 'Y': 1})

    # ── Encoding: nominal — One Hot Encoding ─────
    # region has many categories with no natural order
    df = pd.get_dummies(df, columns=['region'], prefix='region', drop_first=True, dtype=int)

    # ── Outlier detection ─────────────────────────
    df = cap_outliers_iqr(df, 'studied_credits', lower=False)
    # num_of_prev_attempts: max=6, realistic — keep but cap extreme
    df = cap_outliers_iqr(df, 'num_of_prev_attempts', lower=False)

    print(f"\n[studentInfo] Final shape: {df.shape}")
    print(f"Remaining nulls:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


# ═══════════════════════════════════════════════
# 2. studentRegistration.csv
# ═══════════════════════════════════════════════
def process_student_registration():
    df = pd.read_csv(r'C:\Users\kordo\Downloads\archive\studentRegistration.csv', na_values=hidden_nulls)
    print(f"\n{'='*50}\nLoaded: studentRegistration.csv → shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")

    # ── Missing values ──────────────────────────
    # date_unregistration: missing means student NEVER unregistered
    # This is MNAR — the value itself explains the missingness
    # Flag it, then fill with 0 (no unregistration)
    df['unregistered'] = df['date_unregistration'].notnull().astype(int)
    df['date_unregistration'] = df['date_unregistration'].fillna(0)

    # date_registration: if missing, fill with median per module
    if df['date_registration'].isnull().sum() > 0:
        df['date_registration'] = df.groupby('code_module')['date_registration'].transform(
            lambda x: x.fillna(x.median())
        )

    # ── Feature engineering ──────────────────────
    # How long was student registered before unregistering?
    # 0 means never unregistered (stays enrolled)
    df['days_registered'] = df.apply(
        lambda row: row['date_unregistration'] - row['date_registration']
        if row['unregistered'] == 1 else 0, axis=1
    )

    print(f"\n[studentRegistration] Final shape: {df.shape}")
    print(f"Remaining nulls:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


# ═══════════════════════════════════════════════
# 3. assessments.csv
# ═══════════════════════════════════════════════
def process_assessments():
    df = pd.read_csv(r'C:\Users\kordo\Downloads\archive\assessments.csv', na_values=hidden_nulls)
    print(f"\n{'='*50}\nLoaded: assessments.csv → shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")

    # ── Missing values ──────────────────────────
    # date: MNAR — missing only for Exams (100% weight)
    # Final exams have no scheduled date by design
    df['has_date'] = df['date'].notnull().astype(int)
    df['date'] = df['date'].fillna(-1)  # -1 = no fixed date (final exam)

    # weight: fill missing with median per assessment type
    if df['weight'].isnull().sum() > 0:
        df['weight'] = df.groupby('assessment_type')['weight'].transform(
            lambda x: x.fillna(x.median())
        )

    # ── Encoding: assessment_type ─────────────────
    # TMA=coursework, CMA=computer-marked, Exam=final
    type_map = {'TMA': 0, 'CMA': 1, 'Exam': 2}
    df['assessment_type_encoded'] = df['assessment_type'].map(type_map)

    print(f"\n[assessments] Final shape: {df.shape}")
    print(f"Remaining nulls:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


# ═══════════════════════════════════════════════
# 4. studentAssessment.csv
# ═══════════════════════════════════════════════
def process_student_assessment():
    df = pd.read_csv(r'C:\Users\kordo\Downloads\archive\studentAssessment.csv', na_values=hidden_nulls)
    print(f"\n{'='*50}\nLoaded: studentAssessment.csv → shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")

    # ── Missing values ──────────────────────────
    # score: students who submitted (have date_submitted) but no score
    # This is a data entry error — drop these rows
    before = df.shape[0]
    df.dropna(subset=['score'], inplace=True)
    after = df.shape[0]
    print(f"\n[studentAssessment] Dropped {before - after} rows with missing score")

    # ── Outlier detection ─────────────────────────
    # Score is 0-100 — values outside this range are errors
    df['score'] = df['score'].clip(lower=0, upper=100)

    # ── Feature engineering per student ──────────
    # Aggregate to get one row per student
    student_stats = df.groupby('id_student').agg(
        avg_score=('score', 'mean'),          # average score across all assessments
        max_score=('score', 'max'),           # best score
        min_score=('score', 'min'),           # worst score
        total_assessments=('score', 'count'), # how many assessments taken
        avg_days_submitted=('date_submitted', 'mean'),  # avg submission timing
        banked_count=('is_banked', 'sum')     # how many scores were carried over
    ).reset_index()

    # Submission delay: negative = submitted early, positive = submitted late
    # (already in the raw date_submitted column as days relative to assessment date)

    print(f"\n[studentAssessment] Aggregated shape: {student_stats.shape}")
    return df, student_stats


# ═══════════════════════════════════════════════
# 5. vle.csv
# ═══════════════════════════════════════════════
def process_vle():
    df = pd.read_csv(r'C:\Users\kordo\Downloads\archive\vle.csv', na_values=hidden_nulls)
    print(f"\n{'='*50}\nLoaded: vle.csv → shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")

    # ── Missing values ──────────────────────────
    # week_from / week_to: when the resource is available
    # Missing means the resource is available all semester
    df['week_from'] = df['week_from'].fillna(0)
    df['week_to'] = df['week_to'].fillna(df['week_to'].max())

    print(f"\n[vle] Final shape: {df.shape}")
    return df


# ═══════════════════════════════════════════════
# 6. studentVle.csv
# ═══════════════════════════════════════════════
def process_student_vle():
    # studentVle is split into 8 files — read all and combine
    parts = []
    for i in range(8):
        path = rf'C:\Users\kordo\Downloads\archive\studentVle_{i}.csv'
        part = pd.read_csv(path, na_values=hidden_nulls)
        parts.append(part)
        print(f"  Loaded studentVle_{i}.csv → {part.shape[0]} rows")

    df = pd.concat(parts, ignore_index=True)
    print(f"\n{'='*50}\nCombined studentVle → shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")

    # studentVle is typically clean — no missing values
    # But it's very large, so we aggregate immediately

    # ── Outlier detection ─────────────────────────
    # sum_click: IQR method is too aggressive here (most clicks are 1-3)
    # Use 99th percentile instead to only remove extreme values
    upper_99 = df['sum_click'].quantile(0.99)
    df['sum_click'] = df['sum_click'].clip(upper=upper_99)
    print(f"\n[Outlier capping] 'sum_click' capped at 99th percentile: {upper_99}")

    # ── Aggregation per student ───────────────────
    student_vle_stats = df.groupby('id_student').agg(
        total_clicks=('sum_click', 'sum'),      # total engagement
        avg_clicks_per_day=('sum_click', 'mean'), # average daily engagement
        active_days=('date', 'nunique'),         # how many days they were active
        first_activity=('date', 'min'),          # when they first engaged
        last_activity=('date', 'max')            # when they last engaged
    ).reset_index()

    # ── Feature engineering ──────────────────────
    # Activity span: difference between first and last activity
    student_vle_stats['activity_span'] = (
        student_vle_stats['last_activity'] - student_vle_stats['first_activity']
    )

    print(f"\n[studentVle] Aggregated shape: {student_vle_stats.shape}")
    return df, student_vle_stats


# ═══════════════════════════════════════════════
# 7. courses.csv
# ═══════════════════════════════════════════════
def process_courses():
    df = pd.read_csv(r'C:\Users\kordo\Downloads\archive\courses.csv', na_values=hidden_nulls)
    print(f"\n{'='*50}\nLoaded: courses.csv → shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    # courses.csv is typically clean and small
    # module_presentation_length is the only numerical column
    print(f"\n[courses] Final shape: {df.shape}")
    return df


# ═══════════════════════════════════════════════
# MERGING — Build master dataframe
# ═══════════════════════════════════════════════
def merge_all(df_info, df_reg, df_assess_agg, df_vle_agg, df_courses):
    """
    Merge all processed tables into one master dataframe.
    Central key: id_student + code_module + code_presentation
    """
    print("\n" + "="*50)
    print("MERGING ALL TABLES")

    # Start with studentInfo as base
    master = df_info.copy()
    print(f"Base (studentInfo): {master.shape}")

    # Merge registration info
    master = master.merge(
        df_reg[['id_student', 'code_module', 'code_presentation',
                'date_registration', 'unregistered', 'days_registered']],
        on=['id_student', 'code_module', 'code_presentation'],
        how='left'
    )
    print(f"After registration merge: {master.shape}")

    # Merge assessment aggregates (by student only)
    master = master.merge(
        df_assess_agg,
        on='id_student',
        how='left'
    )
    print(f"After assessment merge: {master.shape}")

    # Merge VLE aggregates (by student only)
    master = master.merge(
        df_vle_agg,
        on='id_student',
        how='left'
    )
    print(f"After VLE merge: {master.shape}")

    # Merge course info
    master = master.merge(
        df_courses,
        on=['code_module', 'code_presentation'],
        how='left'
    )
    print(f"After courses merge: {master.shape}")

    # Fill any new nulls created by left joins
    # (students with no VLE activity or no assessments)
    vle_cols = ['total_clicks', 'avg_clicks_per_day', 'active_days',
                'activity_span', 'first_activity', 'last_activity']
    assess_cols = ['avg_score', 'max_score', 'min_score',
                   'total_assessments', 'avg_days_submitted', 'banked_count']

    for col in vle_cols + assess_cols:
        if col in master.columns:
            master[col] = master[col].fillna(0)

    print(f"\nFinal master shape: {master.shape}")
    print(f"Remaining nulls:\n{master.isnull().sum()[master.isnull().sum() > 0]}")
    return master


# ═══════════════════════════════════════════════
# SCALING
# ═══════════════════════════════════════════════
def scale_features(master):
    """
    Scale numerical features to [0, 1] range using MinMaxScaler.
    We do NOT scale:
      - Binary columns (already 0/1)
      - Encoded ordinal columns (already meaningful integers)
      - The target column (final_result)
    """
    cols_to_scale = [
        'studied_credits', 'num_of_prev_attempts', 'imd_band',
        'total_clicks', 'avg_clicks_per_day', 'active_days', 'activity_span',
        'avg_score', 'max_score', 'min_score', 'total_assessments',
        'avg_days_submitted', 'date_registration', 'days_registered',
        'module_presentation_length'
    ]

    # Only scale columns that actually exist in master
    cols_to_scale = [c for c in cols_to_scale if c in master.columns]

    scaler = MinMaxScaler()
    master[cols_to_scale] = scaler.fit_transform(master[cols_to_scale])

    print(f"\n[Scaling] Scaled {len(cols_to_scale)} columns with MinMaxScaler")
    return master


# ═══════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════
def engineer_features(master):
    """
    Create new meaningful features from existing ones.
    Good features = better models.
    """

    # Engagement rate: how active was the student relative to credits taken?
    if 'total_clicks' in master.columns and 'studied_credits' in master.columns:
        master['engagement_rate'] = master['total_clicks'] / (master['studied_credits'] + 1)

    # Performance consistency: difference between best and worst score
    if 'max_score' in master.columns and 'min_score' in master.columns:
        master['score_range'] = master['max_score'] - master['min_score']

    # Assessment completion rate: did they take all assessments?
    if 'total_assessments' in master.columns:
        master['completed_assessments_flag'] = (
            master['total_assessments'] > 0
        ).astype(int)

    # Struggle flag: student who failed before + took many credits
    if 'num_of_prev_attempts' in master.columns and 'studied_credits' in master.columns:
        master['struggle_flag'] = (
            (master['num_of_prev_attempts'] > 0) &
            (master['studied_credits'] > master['studied_credits'].median())
        ).astype(int)

    # Early engagement: did the student start engaging early?
    if 'first_activity' in master.columns:
        master['early_engager'] = (master['first_activity'] < 30).astype(int)

    print(f"\n[Feature Engineering] Master shape after new features: {master.shape}")
    return master


# ═══════════════════════════════════════════════
# MAIN — Run full pipeline
# ═══════════════════════════════════════════════
if __name__ == '__main__':

    print("OULAD FULL PREPROCESSING PIPELINE")
    print("="*50)

    # Step 1 — Load and clean each file
    df_info        = process_student_info()
    df_reg         = process_student_registration()
    df_assessments = process_assessments()
    df_sa_raw, df_sa_agg = process_student_assessment()
    df_vle_raw, df_vle_agg = process_student_vle()
    df_courses     = process_courses()

    # Step 2 — Merge everything
    master = merge_all(df_info, df_reg, df_sa_agg, df_vle_agg, df_courses)

    # Step 3 — Scale numerical features
    master = scale_features(master)

    # Step 4 — Engineer new features
    master = engineer_features(master)

    # Step 5 — Save outputs
    master.to_csv(r'C:\Users\kordo\Downloads\master_clean.csv', index=False)
    df_info.to_csv(r'C:\Users\kordo\Downloads\studentInfo_clean.csv', index=False)
    df_reg.to_csv(r'C:\Users\kordo\Downloads\studentRegistration_clean.csv', index=False)
    df_assessments.to_csv(r'C:\Users\kordo\Downloads\assessments_clean.csv', index=False)
    df_sa_raw.to_csv(r'C:\Users\kordo\Downloads\studentAssessment_clean.csv', index=False)
    df_sa_agg.to_csv(r'C:\Users\kordo\Downloads\studentAssessment_aggregated.csv', index=False)
    df_vle_agg.to_csv(r'C:\Users\kordo\Downloads\studentVle_aggregated.csv', index=False)
    df_courses.to_csv(r'C:\Users\kordo\Downloads\courses_clean.csv', index=False)

    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print(f"Master dataframe: {master.shape}")
    print(f"Columns: {list(master.columns)}")
    print("\nAll files saved to Downloads")
