import pandas as pd

# import numpy as np

data = {"name": ["alice", "bob", "eve", "alice", "bob", "eve", "alice"],
        "col_2": [1, 2, 3, 3, 3, 3, 1],
        "age": [20, 30, 40, 1, 2, 3, 20],
        "y": [0, 1, 2, 2, 1, 2, 0]}

df = pd.DataFrame(data)

print(df)

print(".describe() the data in terms of counts, mean, std dev, percentiles, min, max")
print(df.describe())

print("filter records")
print(type(df[df["name"] == "alice"]))
print(df[df["name"] == "alice"])

print("filter records using .column_name")
print(type(df[df.name == "alice"]))
print(df[df.name == "alice"])

print("filter records by column index")
print(type(df[df[df.columns[0]] == "alice"]))
print(df[df[df.columns[0]] == "alice"])

print("filter records on multiple fields requires '&' and '()'")
print(df[(df["name"] == "alice") & (df["age"] == 20)])

print("get a column or row (pandas data series) using .iloc")
print(type(df.iloc[:, 0]))
print(df.iloc[:, 0])

print("do we have any duplicate input-output vectors?")
print(df.value_counts())

print("what are the .unique values of the column 'name'?")
print(df["name"].unique())

print("what are the values and counts .value_counts() in each column?")
# for col_name in df.columns:
for col_name in df.columns:
    print()
    print(type(col_name), col_name)
    print(df[col_name].value_counts())
    print(type(df[col_name].value_counts()))

print("sort_values a dataframe based on , but not in 'ascending' order")
df_sorted = df.sort_values(["col_2", "age"], ascending=False)
print(df_sorted)

print("what if we want to do joins between two dataframes like between two tables in SQL")
print("let's use .merge with how=left")
df_out = pd.merge(df, df_sorted[df_sorted["name"] == "alice"]["name"], how="left")
print(df_out)

print("what about .join?")

print("explore pandas .transform")
