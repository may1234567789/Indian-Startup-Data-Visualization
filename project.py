
# 0   Sr No             891 non-null    int64
#  1   Date              891 non-null    datetime64[us]
#  2   Startup Name      891 non-null    str
#  3   IndustryVertical  891 non-null    str
#  4   SubVertical       891 non-null    str
#  5   CityLocation      891 non-null    str
#  6   Investors Name    891 non-null    str
#  7   InvestmentType    891 non-null    str
#  8   Amount in USD     891 non-null    int6

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("startup_funding.csv")

# Standardize CSV headers so the rest of the code can use cleaner names.
df = df.rename(columns={
    "Date dd/mm/yyyy": "Date",
    "Industry Vertical": "IndustryVertical",
    "City  Location": "CityLocation",
    "InvestmentnType": "InvestmentType"
})

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df = df.drop("Remarks",axis=1)

print(df.isnull().sum())

# df["Amount in USD"] = df["Amount in USD"].str.replace(",", "").str.strip(),errors=("coerce").fillna(0).astype(float)
df["Amount in USD"] = pd.to_numeric(df["Amount in USD"].str.replace(",", "").str.strip(),errors="coerce").fillna(0).astype(int)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.dropna(subset=["CityLocation", "IndustryVertical", "SubVertical", "InvestmentType", "Amount in USD", "Date","Investors Name"])
print(df.isnull().sum())
print(df.info())

# Year wise funding trend
df["Year"] = df["Date"].dt.year
df_yearly = df.groupby("Year",as_index=False)["Amount in USD"].sum()

plt.bar(df_yearly["Year"], df_yearly["Amount in USD"])
plt.xlabel("Year")
plt.ylabel("Total Funding")
plt.title("Year-wise Total Funding")
plt.show()

# Cities deal Wise
top_cities = df["CityLocation"].value_counts().head(10).index
df_top_cities = df[df["CityLocation"].isin(top_cities)]

plt.figure(figsize=(10, 6))
sns.countplot(data=df_top_cities, x="CityLocation", order=top_cities, color="orange")
plt.xlabel("City")
plt.ylabel("Number of Offers")
plt.title("Top 10 Cities by Number of Offers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Funding category Wise

