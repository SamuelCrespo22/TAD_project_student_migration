import pandas as pd
from ydata_profiling import ProfileReport
import glob
import os

excel_files = glob.glob('data/*.xlsx')

if not excel_files:
    print("No Excel files found in the 'data/' folder.")
else:
    print(f"Found {len(excel_files)} Excel files. Starting processing...")

    for file_path in excel_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        html_output = os.path.join('reports', f"report_{file_name}.html")
        
        print(f"\n--- Processing: {file_name} ---")
        
        print(f"Reading file {file_path} (all rows)...")
        df = pd.read_excel(file_path, engine='openpyxl')

        for col in ['Receiving City', 'Receiving Organization']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()

        print("Generating profiling report (this may take a few minutes)...")
        profile = ProfileReport(df, title=f"Exploratory Analysis - {file_name}", minimal=True)

        profile.to_file(html_output)
        print(f"Report generated successfully! Saved as '{html_output}'")

    print("\nAll reports generated successfully!")