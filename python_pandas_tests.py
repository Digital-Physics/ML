import pandas as pd

data = {"name": ["alice", "bob", "eve", "alice", "bob", "eve"],
        "col_2": [1, 2, 3, 3, 3, 3],
        "age": [20, 30, 40, 1, 2, 3],
        "y": [0, 1, 2, 2, 1, 2]}

df = pd.DataFrame(data)

print(df)

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
