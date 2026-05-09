import pandas as pd
from ydata_profiling import ProfileReport
 
BASE = r'C:\Users\kordo\Downloads\archive'
OUT  = r'C:\Users\kordo\Desktop'
 
hidden_nulls = ['N/A', 'n/a', 'NA', 'na', 'unknown', 'Unknown',
                '?', '', ' ', 'none', 'None', 'NULL', 'null']
 
def report(df, name):
    print(f"Generating report: {name} — shape: {df.shape}")
    profile = ProfileReport(df, title=f"{name} Report", explorative=True)
    profile.to_file(rf"{OUT}\{name}_report.html")
    print(f"Saved: {name}_report.html\n")
 
 
# ── 1. studentInfo ───────────────────────────
df = pd.read_csv(rf'{BASE}\studentInfo.csv', na_values=hidden_nulls)
report(df, 'studentInfo')
 
# ── 2. studentRegistration ───────────────────
df = pd.read_csv(rf'{BASE}\studentRegistration.csv', na_values=hidden_nulls)
report(df, 'studentRegistration')
 
# ── 3. assessments ───────────────────────────
df = pd.read_csv(rf'{BASE}\assessments.csv', na_values=hidden_nulls)
report(df, 'assessments')
 
# ── 4. studentAssessment ─────────────────────
df = pd.read_csv(rf'{BASE}\studentAssessment.csv', na_values=hidden_nulls)
report(df, 'studentAssessment')
 
# ── 5. vle ───────────────────────────────────
df = pd.read_csv(rf'{BASE}\vle.csv', na_values=hidden_nulls)
report(df, 'vle')
 
# ── 6. courses ───────────────────────────────
df = pd.read_csv(rf'{BASE}\courses.csv', na_values=hidden_nulls)
report(df, 'courses')
 
# ── 7. studentVle (8 split files → combine) ──
parts = []
for i in range(8):
    part = pd.read_csv(rf'{BASE}\studentVle_{i}.csv', na_values=hidden_nulls)
    parts.append(part)
    print(f"  Loaded studentVle_{i}.csv → {part.shape[0]:,} rows")
 
df = pd.concat(parts, ignore_index=True)
print(f"Combined studentVle → {df.shape[0]:,} rows")
 
# studentVle is huge — use minimal mode to avoid running out of memory
profile = ProfileReport(df, title="studentVle Report", minimal=True)
profile.to_file(rf"{OUT}\studentVle_report.html")
print("Saved: studentVle_report.html\n")
 

print("ALL REPORTS DONE — check your Desktop")
