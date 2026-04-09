
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
import numpy as np

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

# df = df.drop("Remarks",axis=1)

print(df.isnull().sum())

# df["Amount in USD"] = df["Amount in USD"].str.replace(",", "").str.strip(),errors=("coerce").fillna(0).astype(float)
df["Amount in USD"] = pd.to_numeric(df["Amount in USD"].str.replace(",", "").str.strip(),errors="coerce").fillna(0).astype(int)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.dropna(subset=["CityLocation", "IndustryVertical", "SubVertical", "InvestmentType", "Amount in USD", "Date","Investors Name"])
print(df.isnull().sum())
print(df.info())



# Year wise funding trend
df["Year"] = df["Date"].dt.year
df_yearly = df.groupby("Year", as_index=False)["Amount in USD"].sum()

plt.bar(df_yearly["Year"], df_yearly["Amount in USD"])
plt.xlabel("Year")
plt.ylabel("Total Funding")
plt.title("Year-wise Total Funding")
plt.show()



# Funding category Wise
df["IndustryVertical"] = df["IndustryVertical"].str.strip()
df_category = df.groupby("IndustryVertical", as_index=False)["Amount in USD"].sum().sort_values("Amount in USD", ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=df_category, x="IndustryVertical", y="Amount in USD")
plt.xlabel("Funding Category")
plt.ylabel("Total Funding (USD)")
plt.title("Industry-wise Total Funding")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Investment type Wise
df["InvestmentType"] = df["InvestmentType"].str.strip()
df_investment_type = df.groupby("InvestmentType", as_index=False)["Amount in USD"].sum().sort_values("Amount in USD", ascending=False).head(10)


plt.figure(figsize=(12, 7))
sns.barplot(data=df_investment_type, x="Amount in USD", y="InvestmentType", palette="mako")
plt.xlabel("Total Funding (USD)")
plt.ylabel("Investment Type")
plt.title("Top 10 Investment Types by Funding Volume")
plt.tight_layout()
plt.show()

# Top funded startups

df_top_startups =df.groupby("Startup Name", as_index=False)["Amount in USD"].sum().sort_values("Amount in USD", ascending=False).head(10)


plt.figure(figsize=(12, 7))
sns.barplot(data=df_top_startups, x="Amount in USD", y="Startup Name", palette="viridis")
plt.xlabel("Total Funding Received (USD)")
plt.ylabel("Startup")
plt.title("Top 10 Startups by Total Funding Received")
plt.tight_layout()
plt.show()

# Top investors by funding volume

df_top_investors = df.groupby("Investors Name", as_index=False)["Amount in USD"].sum().sort_values("Amount in USD", ascending=False).head(10)


plt.figure(figsize=(12, 7))
sns.barplot(data=df_top_investors, x="Amount in USD", y="Investors Name", palette="rocket")
plt.xlabel("Total Funding Invested (USD)")
plt.ylabel("Investor")
plt.title("Top 10 Investors by Funding Volume")
plt.tight_layout()
plt.show()

