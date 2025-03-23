# FCC
# Book Recommendation System

## Overview
This project builds a book recommendation system using the Book-Crossings dataset. The system leverages user ratings and a k-nearest neighbors (KNN) algorithm to suggest books similar to a given title based on cosine similarity.

This project is part of the **FreeCodeCamp (FCC) Data Science Course**.

## Dataset
The dataset consists of two CSV files:
1. **BX-Books.csv** - Contains book details such as ISBN, title, and author.
2. **BX-Book-Ratings.csv** - Contains user ratings for books.

The dataset was sourced from [FreeCodeCamp](https://cdn.freecodecamp.org/project-data/books/book-crossings.zip).

## Installation and Setup
1. Download and extract the dataset:
    ```bash
    !wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip
    !unzip book-crossings.zip
    ```
2. Install required Python libraries:
    ```bash
    pip install numpy pandas matplotlib seaborn plotly scikit-learn
    ```

## Data Processing
- Loaded data using `pandas`.
- Handled missing values and cleaned the dataset.
- Filtered users who have rated more than 200 books.
- Filtered books that have more than 100 ratings.
- Created a pivot table where rows represent books, columns represent users, and values represent ratings.

## Exploratory Data Analysis
- Checked for missing values.
- Analyzed distribution of ratings.
- Visualized user rating counts.

## Model Implementation
- Used the `NearestNeighbors` model from `sklearn` with the cosine similarity metric.
- Trained the model on the user-item matrix.
- Implemented a recommendation function to find the top 5 most similar books to a given title.

## Features
- **Book Similarity Search**: Find books similar to a given title.
- **Data Filtering**: Consider only frequently rated books and active users.
- **Visualization**: Show rating distributions and book similarities.

## Example Usage
- Retrieve book recommendations based on a given title.
- Visualize books with the most and least similarity.

## Repository Structure
- `README.md` - This documentation file.
- `data/` - Contains the dataset files.
- `notebooks/` - Jupyter notebooks with data exploration and model training.
- `plots/` - Visualizations generated from the analysis.

## Conclusion
This project successfully implements a content-based book recommendation system using KNN and cosine similarity. It allows users to find similar books based on user ratings.

## References
- [FreeCodeCamp Book-Crossings Dataset](https://cdn.freecodecamp.org/project-data/books/book-crossings.zip)
- [Scikit-learn NearestNeighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Plotly Documentation](https://plotly.com/python/)

