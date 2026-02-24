import pandas as pd

# Configure pandas to display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

file_path = 'Erasmus-KA1-Mobility-Data-2022.xlsx'
df = pd.read_excel(file_path)

# Display basic information about the data
print("Data shape:", df.shape)

print("\nPrimeiras 10 linhas (mostrando todas as colunas):")
print(df.head(10))

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nContagem de '-' por coluna:")
print((df == '-').sum())
