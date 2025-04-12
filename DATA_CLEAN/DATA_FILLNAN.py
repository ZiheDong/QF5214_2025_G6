# Name:Cao Jiayue

import os
import pandas as pd

# === Configuration ===
input_dir = './Data_Clean'
output_dir = './Data_Clean_Filled'  # New Output Folder

os.makedirs(output_dir, exist_ok=True)

# === Process All CSV Files ===
for file in os.listdir(input_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)

        # Fill All NaN With 0
        df.iloc[:, 3] = df.iloc[:, 3].fillna(0)

        # Save To New Folder
        output_path = os.path.join(output_dir, file)
        df.to_csv(output_path, index=False)
        print(f"✔ Finished: {file}")

print("=== All CSV Files Have Been Filled And Saved ===")
