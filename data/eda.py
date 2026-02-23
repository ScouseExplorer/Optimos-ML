import pandas as pd 
import numpy as np

from pathlib import Path

csv_path = Path(__file__).parent / "linkedin_posts.csv"
if not csv_path.exists():
	raise FileNotFoundError(f"CSV file not found at {csv_path.resolve()}; place linkedin_posts.csv in the same folder as this script or provide the correct path")
df = pd.read_csv(csv_path)

print(df.head(1))