# Name: Gao Xinyu,

import openai
import pandas as pd
import time

openai.api_key = "sk-API key"

# ===== Function GPT Model =====
def get_ai_weight(variable, meaning, retries=3, delay=1):
    prompt = f"""
You are an expert in agricultural environmental evaluation. Based on the variable name and its explanation, determine:
- Whether it has a positive or negative impact on agricultural environmental quality
- An importance weight between 0 and 1 (float)

Respond in the format:
Direction: Positive/Negative
Weight: [float number]

Variable: {variable}
Meaning: {meaning}
"""

    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message["content"]
        except Exception as e:
            time.sleep(delay)
    return "Error: Failed after retries"

# ===== GPT Response =====
def parse_result(result):
    try:
        direction = "Positive" if "Positive" in result else "Negative"
        weight = float(result.split("Weight:")[-1].strip())
        return pd.Series([direction, weight])
    except:
        return pd.Series(["Unknown", 0.0])

# ===== Main =====
def main():
    input_file = "Index_Combined.csv"  # contain 'variable' and 'variable_meaning'
    output_file = "AI_Assigned_Index_Weight.csv"

    df = pd.read_csv(input_file)
    results = []

    for i, row in df.iterrows():
        print(f"Processing {i+1}/{len(df)}: {row['variable']}")
        result = get_ai_weight(row['variable'], row['variable_meaning'])
        results.append(result)
        time.sleep(1.5)

    df['AI_Result'] = results
    df[['AI_Judged_Impact', 'AI_Assigned_Weight']] = df['AI_Result'].apply(parse_result)

    df.to_csv(output_file, index=False)
    print(f"✅ Output saved to {output_file}")

if __name__ == "__main__":
    main()
