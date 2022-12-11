# %%
import pandas as pd

INPUT_FILE = "dataset/books.csv"

# %%
# Encoding determined by opening file in a text editor
with open(INPUT_FILE, "r", encoding='utf-8') as inputFile:
    firstFiveLines = [next(inputFile) for line in range(5)]
    fileLineCount = sum(1 for _ in inputFile) + 5
print(firstFiveLines)

# %%
rawDF = pd.read_csv(INPUT_FILE, sep=",", index_col="bookID",
                    on_bad_lines="skip", encoding="utf-8")
print(rawDF.head())

# %%
print("Raw file line count (excluding header): " + str(fileLineCount-1))
print("Imported Dataframe shape: " + str(rawDF.shape))

# %%
print(rawDF.dtypes)

# %%
# dropping "isbn" column as "isbn13" better conveys the same information
rawDF = rawDF.drop(columns=["isbn"])

# %%
# typical null check returns no nulls, but needs further checks for "object" columns
print(rawDF.isna().sum())

# %%
objectColumns = rawDF.select_dtypes(include=["object"]).columns
for column in objectColumns:
    print("%s: %s"%(column, str(sum(rawDF[column].values == ""))))

# %%
