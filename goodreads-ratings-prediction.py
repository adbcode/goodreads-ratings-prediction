# %%
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, max_error, explained_variance_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.tree import DecisionTreeRegressor

INPUT_FILE = "dataset/books.csv"

np.random.seed(42)

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

# # %%
# # explore potentially categorical values

# # %%
# print(df.loc[df.language_code.str.contains("-")].language_code.value_counts())

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

# # %%
# # removing books with num_pages < 25
# df = df[df["num_pages"] > 25]
# print(df.shape)

# # %%
# # checking books with high page count (95th percentile rounded to nearest tenth)
# print(df.loc[df["num_pages"] > 750].shape)
# # exploring low page counts
# df.loc[df["num_pages"] > 750].num_pages.plot.box()

# # %%
# # despite the high count and range, removing the top 5% num_pages
# df = df[df["num_pages"] <= 750]
# print(df.shape)
# df.num_pages.plot.box()

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
# drop title (and main_title)
df.drop(["title"], axis=1, inplace=True) #df.drop(["main_title", "title"], axis=1, inplace=True)
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
# fix 0-page books to prevent calculation issues
df["num_pages"] = df.num_pages.apply(lambda x: 1 if x == 0 else x)
# generate features using existing numerical ones
df["text_reviews_ratio"] = df["text_reviews_count"] / df["ratings_count"]
df["text_reviews_count_per_page"] = df["text_reviews_count"] / df["num_pages"]
df["ratings_count_per_page"] = df["ratings_count"] / df["num_pages"]
# artificial attributes that might help
df["ratings_count_times_text_reviews_count"] = df["text_reviews_count"] * df["ratings_count"]
df["text_reviews_count_times_num_pages"] = df["text_reviews_count"] * df["num_pages"]
df["ratings_count_times_num_pages"] = df["ratings_count"] * df["num_pages"]
df["ratings_count_times_text_reviews_count_times_num_pages"] = df["text_reviews_count"] * df["ratings_count"] * df["num_pages"]

# %%
# categorical label-level aggregation features

main_author_mean = df.groupby("main_author").mean()
main_author_count = df.groupby("main_author").count()
publisher_count = df.groupby("publisher").count()
publication_year_sum = df.groupby("publication_year").sum()
publication_month_sum = df.groupby("publication_month").sum()

df["english_book"] = df.language_code.apply(lambda x: 1 if x.startswith("eng") else 0)
df["main_author_book_count"] = df.main_author.apply(lambda x: main_author_count["isbn13"][x])
df["main_author_average_text_reviews_count"] = df.main_author.apply(lambda x: main_author_mean["ratings_count"][x])
df["main_author_average_text_reviews_count"] = df.main_author.apply(lambda x: main_author_mean["text_reviews_count"][x])
df["publisher_count"] = df.publisher.apply(lambda x: publisher_count["isbn13"][x])
df["publication_year_ratings_count"] = df.publication_year.apply(lambda x: publication_year_sum["ratings_count"][x])
df["publication_year_text_reviews_count"] = df.publication_year.apply(lambda x: publication_year_sum["text_reviews_count"][x])
df["publication_month_ratings_count"] = df.publication_month.apply(lambda x: publication_month_sum["ratings_count"][x])
df["publication_month_text_reviews_count"] = df.publication_month.apply(lambda x: publication_month_sum["text_reviews_count"][x])

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
language_code_counts = rawDF.language_code.value_counts()
print(language_code_counts)
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
# drop authors and main_author
df.drop(["main_author", "authors"], axis=1, inplace=True)
# df.drop(["publisher", "language_code"], axis=1, inplace=True)
df.drop(["main_title"], axis=1, inplace=True)
print(df.head())

# %%
# feature selection
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

get_correlation_matrix_graph(df.corr())

# %%
# new synthetic features
df["mean_title_word_length"] = df["title_length"] / df["title_word_count"]
df["publisher_PenguinBooks_times_count"] = df["publisher_count"] * df["publisher_Vintage"]
df["publisher_Vintage_times_count"] = df["publisher_Vintage"] * df["publisher_count"]

# %%
# scatterplots of each feature with average_rating
def get_scatterplot_wrt_target(df, target_feature, feature):
    test = df[[feature, target_feature]].groupby([feature, target_feature]).value_counts().reset_index(name="count")
    px.scatter(data_frame=test, x=feature, y=target_feature, color="count").show()

def get_scatterplots_wrt_target_for_df(df, target_feature):
    for feature in df.columns:
        if feature != target_feature and feature in df.select_dtypes(include=[np.number]).columns:
            get_scatterplot_wrt_target(df, target_feature, feature)

get_scatterplots_wrt_target_for_df(df, "average_rating")

# %%
# functions to perform training and scoring in batch

def score_model(regressor, y_test, y_pred, X_test):
    print(f"mean absolute error:\t\t{mean_squared_error(y_true = y_test, y_pred = y_pred)}")
    print(f"max error:\t\t\t{max_error(y_true = y_test, y_pred = y_pred)}")
    print(f"r2 score:\t\t\t{regressor.score(X_test, y_test)}")
    print(f"explained variance score:\t{explained_variance_score(y_true = y_test, y_pred = y_pred)}")

def batch_regression_model_training(df, regressor_list, target_feature):
    y = df[target_feature]
    X = df.drop(target_feature, axis=1)

    # # normalize dataset
    # scaler = MinMaxScaler()
    # X = pd.DataFrame(scaler.fit_transform(X.values), columns=X.columns, index=X.index)
    # # print(X.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset split into {len(y_train)} train rows and {len(y_test)} test rows.")

    # batch training
    for regressor in regressor_list:
        print(regressor)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        score_model(regressor, y_test, y_pred, X_test)

# %%
# define the models to test
regressor_list = [RandomForestRegressor(random_state = 42), AdaBoostRegressor(random_state = 42),
    DecisionTreeRegressor(random_state = 42), LinearRegression(), DummyRegressor()
]

# %%
# run the training
batch_regression_model_training(df, regressor_list, "average_rating")

# %%
# run on original dataset to sanity check
batch_regression_model_training(rawDF.select_dtypes(include=[np.number]), regressor_list, "average_rating")

# %%
# Hyperparameter of RandomForestRegressor
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 19)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 2, 3, 4]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(3, 20, num = 17)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in np.linspace(2, 15, num = 13)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(2, 20, num = 9)]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# %%
rf_tuner = RandomizedSearchCV(estimator = regressor_list[0],
    param_distributions=random_grid, n_iter=100, cv=2,
    scoring="r2", return_train_score=True,
    verbose=2, random_state=42, n_jobs=-1)

y = df["average_rating"]
X = df.drop("average_rating", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# rf_tuner.fit(X_train, y_train)

# print(rf_tuner.best_params_)

# # %%
# tuned_rf = rf_tuner.best_estimator_
# y_pred = tuned_rf.predict(X_test)
# score_model(tuned_rf, y_test, y_pred, X_test)

# %%
# sanity check with the "stock" model
stock_rf = regressor_list[0]
stock_rf.fit(X_train, y_train)
y_pred = stock_rf.predict(X_test)
score_model(stock_rf, y_test, y_pred, X_test)

# %%
# compare prediction by reducing precision
def scores_with_varying_precision(y_pred, y_test):
    # baseline
    print(f"baseline\t{np.sum(y_pred == y_test)}\t{round(np.sum(y_pred == y_test)*100/len(y_pred), 1)}%")

    # to nearest tenth
    y_test_tenth = np.array([round(x, 1) for x in y_test])
    y_pred_tenth = np.array([round(x, 1) for x in y_pred])
    print(f"tenth\t\t{np.sum(y_pred_tenth == y_test_tenth)}\t{round(np.sum(y_pred_tenth == y_test_tenth)*100/len(y_pred_tenth), 1)}%")

    # dummy predict top two values (integers)
    y_test_whole = np.array([round(x) for x in y_test])
    y_pred_dummy = np.random.randint(3, 5, len(y_test))
    print(f"dummy\t\t{np.sum(y_test_whole == y_pred_dummy)}\t{round(np.sum(y_test_whole == y_pred_dummy)*100/len(y_pred_dummy),1)}%")

    # to nearest half
    y_test_half = np.array([round(x * 2) / 2 for x in y_test])
    y_pred_half = np.array([round(x * 2) / 2 for x in y_pred])
    print(f"half\t\t{np.sum(y_test_half == y_pred_half)}\t{round(np.sum(y_test_half == y_pred_half)*100/len(y_pred_half), 1)}%")

    # to nearest number
    y_pred_whole = np.array([round(x) for x in y_pred])
    print(f"number\t\t{np.sum(y_test_whole == y_pred_whole)}\t{round(np.sum(y_test_whole == y_pred_whole)*100/len(y_pred_whole),1)}%")

scores_with_varying_precision(y_pred, y_test)

# %%
# scores prediction visually
# baseline
px.scatter(x=y_test, y=y_pred, range_x=[0,5.1], range_y=[0,5.1], labels={'x':'real', 'y':'predicted'}, trendline="ols").show()

# %%
# whole numbers
y_test_whole = np.array([round(x) for x in y_test])
y_pred_whole = np.array([round(x) for x in y_pred])
px.scatter(x=y_test_whole, y=y_pred_whole, range_x=[0,5.1], range_y=[0,5.1], labels={'x':'real', 'y':'predicted'}).show()

# %%
# getting an idea of imabalance between the target values
print(df.average_rating.value_counts().value_counts())

# %%
# # Undersampling -> SMOTE
# undersampler = RandomUnderSampler(sampling_strategy={4:5000}, random_state=42)
# oversampler = SMOTE(sampling_strategy={2:200, 5:250}, random_state=42)

# y_whole = np.array([round(x) for x in y])

# X_bal, y_bal = undersampler.fit_resample(X, y_whole)
# X_bal, y_bal = oversampler.fit_resample(X_bal, y_bal)

# # %%
# # test with stock random forest
# stock_rf = regressor_list[0]
# X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)
# stock_rf.fit(X_train, y_train)
# y_pred = stock_rf.predict(X_test)
# score_model(stock_rf, y_test, y_pred, X_test)

# %%
# mapping target variable to non-linear functions
y_sqrt = df.average_rating.map(lambda y: np.sqrt(y))
y_log_10 = df.average_rating.map(lambda y: np.log10(y))
y_log_2 = df.average_rating.map(lambda y: np.log2(y))
y_log_e = df.average_rating.map(lambda y: np.log(y))
#y_bc, lambda_bc = stats.boxcox(y)
box_cox_transformer = PowerTransformer(method="box-cox")
y_bc = box_cox_transformer.fit_transform(df.average_rating.array.reshape(-1, 1)).ravel()
y_mapped = [y_bc] # [y_sqrt, y_log_10, y_log_2, y_log_e]

# %%
# test performance
for y in y_mapped:
    stock_rf = regressor_list[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    stock_rf.fit(X_train, y_train)
    y_pred = stock_rf.predict(X_test)
    score_model(stock_rf, y_test, y_pred, X_test)
    scores_with_varying_precision(y_pred, y_test)
    px.scatter(x=y_test, y=y_pred, range_x=[min(min(y_test), min(y_pred)),1.1*max(max(y_test), max(y_pred))], range_y=[min(min(y_test), min(y_pred)),1.1*max(max(y_test), max(y_pred))], labels={'x':'real', 'y':'predicted'}).show()

# %%
# sanity check by running transformation on original datset
stock_rf = regressor_list[0]
X_raw = rawDF.drop("average_rating", axis=1).select_dtypes(include=[np.number])
y_raw_bc = box_cox_transformer.fit_transform(rawDF.average_rating.map(lambda x: 0.01 if x == 0 else x).array.reshape(-1, 1)).ravel()
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw_bc, test_size=0.2, random_state=42)
stock_rf.fit(X_train, y_train)
y_pred = stock_rf.predict(X_test)
score_model(stock_rf, y_test, y_pred, X_test)

# %%