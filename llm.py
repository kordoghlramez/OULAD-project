

from dotenv import load_dotenv
import os
import pandas as pd
from openai import OpenAI

# -----------------------------
# 1. Load API Key (SAFE)
# -----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("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" \
""))


# -----------------------------
# 2. Load Clean Data
# -----------------------------
df = pd.read_csv("master_clean.csv")

# -----------------------------
# 3. Take Sample
# -----------------------------
sample = df.head(20)
data_text = sample.to_string()

# -----------------------------
# 4. Create Prompt
# -----------------------------
prompt = f"""
You are a data analyst working on student learning data.

Here is a sample of the dataset:
{data_text}

Tasks:
1. Identify key patterns
2. What affects student performance?
3. Any data quality issues?
4. Give insights
"""

# -----------------------------
# 5. Call LLM
# -----------------------------
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": prompt}]
)

# -----------------------------
# 6. Output
# -----------------------------
output = response.choices[0].message.content

print("\n=== LLM OUTPUT ===\n")
print(output)

# -----------------------------
# 7. Save Output
# -----------------------------
with open("llm_output.txt", "w", encoding="utf-8") as f:
    f.write(output)

print("\n✅ LLM analysis saved")