import pandas as pd
import numpy as np

df = pd.read_csv('assets/dataset/data.csv')
print(df.shape)
df.dropna(inplace=True)
print(df.shape)

# Trâu