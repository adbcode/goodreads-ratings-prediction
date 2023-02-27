# goodreads-ratings-prediction
An exploration to create a regression model on book ratings in UCSD Book Graph's Goodreads Dataset.

## Rationale
We are presented with a curated dataset of book records from [Goodreads](https://www.goodreads.com/) compiled from data collected for [a UCSD project](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home).

It contains the following features:
1) **bookID**: A unique identification number for each book.
2) **title**: The name under which the book was published.
3) **authors**: The names of the authors of the book. Multiple authors are delimited by “/”.
4) **average_rating**: The average rating of the book received in total.
5) **isbn**: Another unique number to identify the book, known as the International
Standard Book Number.
6) **isbn13**: A 13-digit ISBN to identify the book, instead of the standard 11-digit ISBN.
7) **language_code**: Indicates the primary language of the book. For instance, “eng” is standard for English.
8) **num_pages**: The number of pages the book contains.
9) **ratings_count**: The total number of ratings the book received.
10) **text_reviews_count**: The total number of written text reviews the book received.
11) **publication_date**: The date the book was published.
12) **publisher**: The name of the book publisher.

In this project we test the hypothesis that we can predict the *average_rating* value from the other metadata. We treat this as a regression problem and hence test the following set of models:
- Random Forest
- AdaBoost
- Decision Tree
- Linear Regression

## Results
Given the unbalanced nature of the dataset and its inherent bias, the measure of success determines if the hypothesis is proved to be correct:
- **~91%** accuracy in predicting the nearest number.
- **~66%** accuracy for predicting the nearest half.
- ...
- **~0.17** R<sup>2</sup> score for the best regression model.
    - without any feature engineering results in ~0.13.

As majority of the data is centered around 3 and 4, a dummy model can achieve good results if we reduce the precision for this dataset.

## How to deploy
### Create conda environment
- Run `conda env create -f goodreads-rating.yml`
### Run the project
- Run the file in command line using `python goodreads-ratings-prediction.py` from conda prompt of your environment (should be `goodreads-rating`)