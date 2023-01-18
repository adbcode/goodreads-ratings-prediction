# %%
import numpy as np
import pandas as pd
import plotly.figure_factory as ff

from scipy.stats import zscore

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
# removing spaces to help with feature engineering later
df.publisher = df.publisher.map(lambda x: x.replace(" ", ""))
print(df.publisher.value_counts())

# %%
print(df.loc[
        df.ratings_count == 0,
        ["average_rating"]
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
        ["average_rating"]
    ].value_counts().head())

# %%
# replace missing rating counts with the median
median_ratings_count = df.ratings_count.median()
df.ratings_count = df.ratings_count.map(lambda x: median_ratings_count if x == 0 else x)
print(df.ratings_count.value_counts())

# %%
# check numerical features for outliers
df["num_pages"] = df["  num_pages"]
df.drop("  num_pages", axis=1, inplace=True)

# %%
df.num_pages.plot.box()

# %%
print(df.num_pages.quantile(q=[0.01, 0.95]))
print(df.loc[df["num_pages"] < 25].shape)
# exploring low page counts
df.loc[df["num_pages"] < 25].num_pages.plot.box()

# %%
# removing books with num_pages < 25
df = df[df["num_pages"] > 25]
print(df.shape)

# %%
# checking books with high page count (95th percentile rounded to nearest tenth)
print(df.loc[df["num_pages"] > 750].shape)
# exploring low page counts
df.loc[df["num_pages"] > 750].num_pages.plot.box()

# %%
# despite the high count and range, removing the top 5% num_pages
df = df[df["num_pages"] <= 750]
print(df.shape)
df.num_pages.plot.box()

# %%
# convert publication_date to date dtype and extract features
# df["processed_date"] = pd.to_datetime(df["publication_date"])

# %%
# this fails the first time. checking and fixing the incorrect dates from the logs
df.loc[df["publication_date"] == "11/31/2000", "publication_date"] = "11/30/2000"
df.loc[df["publication_date"] == "6/31/1982", "publication_date"] = "6/30/1982"

# %%
# retrying the above
df["processed_date"] = pd.to_datetime(df["publication_date"])
print(df.head())

# %%
# extracting year and month of publication and dropping the original feature
df["publication_year"] = df["processed_date"].dt.year
df["publication_month"] = df["processed_date"].dt.month
df.drop(["processed_date", "publication_date"], axis=1, inplace=True)
print(df.head())

# %%
# fixing ratings_count's dtype
df["ratings_count"] = df["ratings_count"].astype(int)
print(df.head())

# %%
# extract features from title
# removing extra parts from title (e.g. the sub-title after colon or brackets)
def get_main_title(title):
    main_title = title.split(":")
    main_title = main_title[0]
    main_title = main_title.split("(")
    main_title = main_title[0]
    return main_title

df["main_title"] = df.title.apply(get_main_title)
print(df.loc[:, ["title", "main_title"]])

# %%
# get (main) title length and word count
df["title_length"] = df["main_title"].str.len()
df["title_word_count"] = df["main_title"].apply(lambda x: len(x.split()))
print(df.head())

# %%
# drop title and main_title
df.drop(["main_title", "title"], axis=1, inplace=True)
print(df.head())

# %%
# extract features from authors
# extract total author count and the first author name
def get_author_count(authors):
    author_count = authors.split("/")
    author_count = len(author_count)
    return author_count

def get_main_author(authors):
    main_author = authors.split("/")
    main_author = main_author[0]
    return main_author

# %%
df["author_count"] = df.authors.apply(get_author_count)
df["main_author"] = df.authors.apply(get_main_author)
print(df.head())

# %%
# get main author's name length and word "count"
df["main_author_name_length"] = df["main_author"].str.len()
df["main_author_name_word_count"] = df["main_author"].apply(lambda x: len(x.split()))
df["main_author_short_name_count"] = df["main_author"].str.count("\.")
print(df.head())

# %%
# drop authors and main_author
df.drop(["main_author", "authors"], axis=1, inplace=True)
print(df.head())

# %%
# feature selection
# normalize numerical features
features_to_normalize = ["isbn13", "ratings_count", "text_reviews_count", "num_pages",
    "publication_year", "publication_month", "title_length", "title_word_count", "author_count",
    "main_author_name_length", "main_author_name_word_count", "main_author_short_name_count"
]

for feature in features_to_normalize:
    df[feature] = zscore(df[feature])

print(df.head())

# %%
# one-hot encode categorical features
df = pd.get_dummies(df)
print(df.shape)
print(df.head(1))

# %%
# print correlation matrix for the current dataframe
correlation_to_target = df.corr()['average_rating'][1:]
print(correlation_to_target)

# %%
# drop the features with |correlation| < 0.01
low_correlation = correlation_to_target[(correlation_to_target < 0.01) & (correlation_to_target > -0.01)]
df.drop(low_correlation.index.tolist(), axis=1, inplace=True)
print(df.head())

# %%
# better correlation matrix to quickly visualize
new_correlation_matrix = df.corr()
data = np.array(new_correlation_matrix)
fig = ff.create_annotated_heatmap(
    data,
    x = list(new_correlation_matrix.columns),
    y = list(new_correlation_matrix.index),
    annotation_text = np.around(data, decimals = 2),
    hoverinfo = "z",
    colorscale= "Viridis"
)
fig.show()

# %%
# creating the above functionality as function for ease of use
def get_correlation_matrix_graph(correlation_matrix):
    data = np.array(correlation_matrix)
    fig = ff.create_annotated_heatmap(
        data,
        x = list(correlation_matrix.columns),
        y = list(correlation_matrix.index),
        annotation_text = np.around(data, decimals = 2),
        hoverinfo = "z",
        colorscale= "Viridis"
    )
    return fig

# %%