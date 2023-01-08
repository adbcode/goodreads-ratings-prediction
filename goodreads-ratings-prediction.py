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
print(f"Raw file line count (excluding header): {str(fileLineCount-1)}")
print(f"Imported Dataframe shape: {str(rawDF.shape)}")

# %%
print(rawDF.dtypes)

# %%
# dropping "isbn" column as "isbn13" better conveys the same information
df = rawDF.drop(columns=["isbn"])

# %%
# typical null check returns no nulls, but needs further checks for "object" columns
print(df.isna().sum())

# %%
objectColumns = df.select_dtypes(include=["object"]).columns
for column in objectColumns:
    print(f"{column}: {sum(df[column].values == '')}")

# %%
# explore potentially categorical values
language_code_counts = rawDF.language_code.value_counts()
print(language_code_counts)

# %%
print(df.loc[df.language_code.str.contains("-")].language_code.value_counts())

# %%
rare_languages = language_code_counts.loc[
        (language_code_counts < 50) &
        ~(language_code_counts.index.str.contains("-"))
    ].index.values.tolist()
print(rare_languages)   # excluded eng-... variant, i.e. en-CA

# %%
# replace rare languages with "other" value
df.language_code = df.language_code.map(lambda x: "other" if x in rare_languages else x)
print(df.language_code.value_counts())

# %%
publisher_counts = df.publisher.value_counts()
print(publisher_counts)

# %%
small_publishers = publisher_counts.loc[(publisher_counts < 100)].index.values.tolist()
print(f"Out of {len(publisher_counts)} publishers, {len(small_publishers)} are \"small\".")

# %%
# replace small publishers with "other" value
df.publisher = df.publisher.map(lambda x: "other" if x in small_publishers else x)
print(df.publisher.value_counts())

# %%
print(df.loc[
        df.ratings_count == 0,
        ['average_rating']
    ].value_counts())

# %%
print(df.loc[(df.average_rating == 0) & (df.ratings_count != 0)])

# %%
# removing rows with 0 average ratings (and missing ratings count)
df = df.loc[df.average_rating != 0]
print(df.shape)

# %%
print(df.loc[
        df.ratings_count == 0,
        ['average_rating']
    ].value_counts().head())

# %%
# replace missing rating counts with the median
median_ratings_count = df.ratings_count.median()
df.ratings_count = df.ratings_count.map(lambda x: median_ratings_count if x == 0 else x)
print(df.ratings_count.value_counts())

# %%
