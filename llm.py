import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

# 1. LOAD API KEY
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# 2. LOAD CLEAN CSV FILES
# -----------------------------
assessments = pd.read_csv("assessments_clean.csv")
courses = pd.read_csv("courses_clean.csv")
studentAssessment = pd.read_csv("studentAssessment_clean.csv")
studentInfo = pd.read_csv("studentInfo_clean.csv")
studentRegistration = pd.read_csv("studentRegistration_clean.csv")
studentVle = pd.read_csv("studentVle_aggregated.csv")

# -----------------------------
# 3. CLEAN COLUMN NAMES (avoid hidden spaces)
# -----------------------------
for df_temp in [assessments, courses, studentAssessment, studentInfo, studentRegistration, studentVle]:
    df_temp.columns = df_temp.columns.str.strip()

# -----------------------------
# 4. MERGE DATA (CORRECT ORDER)
# -----------------------------

# STEP 1: get module info first
df = studentAssessment.merge(
    assessments,
    on="id_assessment",
    how="left"
)

# STEP 2: add student info
df = df.merge(
    studentInfo,
    on=["id_student", "code_module", "code_presentation"],
    how="left"
)

# STEP 3: add course info
df = df.merge(
    courses,
    on=["code_module", "code_presentation"],
    how="left"
)

# STEP 4: add registration
df = df.merge(
    studentRegistration,
    on=["id_student", "code_module", "code_presentation"],
    how="left"
)

# STEP 5: add activity
df = df.merge(
    studentVle,
    on="id_student",
    how="left"
)

# -----------------------------
# 5. FINAL CLEAN
# -----------------------------
df = df.drop_duplicates()
df = df.fillna(0)

# -----------------------------
# 6. CREATE SUMMARY
# -----------------------------
summary = df.describe().to_string()
missing = df.isnull().sum().to_string()

# -----------------------------
# 7. CREATE PROMPT
# -----------------------------
prompt = f"""
You are a data analyst working on student learning data.

Dataset Summary:
{summary}

Missing Values:
{missing}

Tasks:
1. Identify key patterns
2. What affects student performance?
3. Any data quality issues?
4. Give insights and recommendations
"""

# -----------------------------
# 8. CALL LLM
# -----------------------------
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": prompt}]
)

# -----------------------------
# 9. OUTPUT
# -----------------------------
output = response.choices[0].message.content

print("\n=== LLM OUTPUT ===\n")
print(output)

# -----------------------------
# 10. SAVE OUTPUT
# -----------------------------
with open("llm_output.txt", "w", encoding="utf-8") as f:
    f.write(output)

print("\n✅ LLM analysis saved")
