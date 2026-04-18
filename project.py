import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error,t2_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# load data
df = pd.read_csv(r"C:\Users\mayan\Downloads\startup_funding.csv")

# basic cleaning
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year

df['Amount in USD'] = df['Amount in USD'].astype(str).str.replace(',', '')
df['Amount in USD'] = pd.to_numeric(df['Amount in USD'], errors='coerce')


df = df.dropna(subset=['Amount in USD', 'Year'])
print(df.columns)
# -----------------------------
# 1. Yearly total funding
# -----------------------------
yearly = df.groupby('Year')['Amount in USD'].sum()

plt.figure(figsize=(12, 7))
plt.plot(yearly.index, yearly.values)
plt.title("Yearly Total Funding")
plt.xlabel("Year")
plt.ylabel("Funding")
plt.show()


# -----------------------------
# 2. Category wise funding
# -----------------------------
category = df.groupby('Industry Vertical')['Amount in USD'].sum()
category = category.sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 7))
sns.barplot(x=category.index, y=category.values, palette="rocket")
plt.title("Top Categories by Funding")
plt.xticks(rotation=45)
plt.show()


# -----------------------------
# 3. Top 10 investors
# -----------------------------
df_top_investors = df.groupby("Investors Name", as_index=False)["Amount in USD"].sum().sort_values("Amount in USD", ascending=False).head(10)


plt.figure(figsize=(12, 7))
sns.barplot(data=df_top_investors, x="Amount in USD", y="Investors Name", palette="bright")
plt.xlabel("Total Funding Invested (USD)")
plt.ylabel("Investor")
plt.title("Top 10 Investors by Funding Volume")
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Funding types
# -----------------------------
fund_type = df['InvestmentnType'].value_counts().head(10)

plt.figure(figsize=(12, 7))
plt.bar(fund_type.index, fund_type.values)
plt.title("Funding Types")
plt.xticks(rotation=45)
plt.show()


df_top_startups =df.groupby("Startup Name", as_index=False)["Amount in USD"].sum().sort_values("Amount in USD", ascending=False).head(10)


plt.figure(figsize=(12, 7))
sns.barplot(data=df_top_startups, x="Amount in USD", y="Startup Name", palette="viridis")
plt.xlabel("Total Funding Received (USD)")
plt.ylabel("Startup")
plt.title("Top 10 Startups by Total Funding Received")
plt.tight_layout()
plt.show()

# 6 Cities with most startups
city_counts = df['City  Location'].value_counts().head(10)
plt.figure(figsize=(12, 7))
sns.barplot(x=city_counts.values, y=city_counts.index, palette="magma")
plt.xlabel("Number of Startups")
plt.ylabel("City")
plt.title("Top 10 Cities with Most Startups")
plt.tight_layout()
plt.show()

# 7. Cities with most funding
city_funding = df.groupby('City  Location')['Amount in USD'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 7))
sns.barplot(x=city_funding.values, y=city_funding.index, palette="coolwarm")
plt.xlabel("Total Funding (USD)")
plt.ylabel("City")
plt.title("Top 10 Cities by Total Funding")
plt.tight_layout()
plt.show()


count_year = df.groupby('Year').size()

x = count_year.index.values
y = count_year.values
x = x.reshape(-1,1)
model = LinearRegression()
model.fit(x,y)
y_pred = model.predict(x)

plt.figure()
plt.plot(x, y)
plt.plot(x, y_pred)
plt.legend(["Actual", "Predicted"])
plt.title("Number of Investments Trend")
plt.show()

mse=mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2=r2_score(y, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)