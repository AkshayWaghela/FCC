# FCC

## 📚 Project Index

1. [Rock-Paper-Scissors AI Strategy Explorer](#1-rock-paper-scissors-ai-strategy-explorer)  
2. [Cats vs Dogs Image Classification](#2-cats-vs-dogs-image-classification)  
3. [Health Insurance Cost Prediction](#3-health-insurance-cost-prediction)  
4. [Book Recommendation System](#4-book-recommendation-system)  
5. [Your Fifth Project Title Here](#5-your-fifth-project-title-here)

---

# 🤖 Rock-Paper-Scissors AI Strategy Explorer

Welcome to the **Rock-Paper-Scissors AI Strategy Explorer** — a project built as part of the [freeCodeCamp Machine Learning with Python Certification](https://www.freecodecamp.org/learn/machine-learning-with-python/) curriculum.

This notebook explores different bot strategies for playing Rock-Paper-Scissors (RPS) and evaluates them through simulation. The highlight of the project is a custom **Markov Chain-based AI** that predicts opponent moves and aims to consistently achieve a **60%+ win rate**.

---

## 📌 Project Goals

- ✅ Implement and simulate various Rock-Paper-Scissors bot strategies.
- ✅ Build a custom AI using a 6-length Markov Chain model.
- ✅ Evaluate and compare bot performance over multiple simulations.
- ✅ Visualize win rates, move frequencies, and win/loss streaks.
- ✅ Create an interactive dashboard for bot matchups.

---

## 🧠 Strategies Implemented

| Bot Name | Strategy Description |
|----------|-----------------------|
| **Quincy**  | Cycles through fixed moves (`R`, `R`, `P`, `P`, `S`) |
| **Mrugesh** | Tracks most frequent opponent move in the last 10 rounds |
| **Kris**    | Reacts to opponent’s last move |
| **Abbey**   | Predicts next move based on 2-length move sequence patterns |
| **Random**  | Chooses moves uniformly at random |
| **My Bot**  | 6-length Markov Chain prediction model |

---

## 📈 Performance Evaluation

The AI was tested against each opponent over multiple 1000-round games. Metrics include:

- **Win Rate (%)**
- **95% Confidence Intervals** for repeated simulations
- **Move Frequency Distributions**
- **Win/Loss Streak Distributions**

Visualizations are created using `matplotlib` and `seaborn`.

---

## 🛠️ Technologies Used

- Python 3
- NumPy, pandas, SciPy
- Matplotlib, Seaborn
- ipywidgets (for interactivity)
- Jupyter Notebook

---

## 📊 Sample Result

<img src="images/win_rate_plot.png" alt="Win rate bar chart" width="600"/>

> Example plot: My Bot consistently beats other bots with a >60% win rate.

---

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rps-ai-strategy-explorer.git
   cd rps-ai-strategy-explorer




# Cats vs Dogs Image Classification with TensorFlow

This project trains a convolutional neural network (CNN) to classify images of cats and dogs using TensorFlow and Keras. The model is trained on a labeled dataset with data augmentation, validated, and tested on an unknown test set.

## Project Overview

- Download and extract the cats and dogs dataset.
- Preprocess images with data augmentation on the training set.
- Build a CNN model for binary classification (cats vs dogs).
- Train the model with early stopping based on validation accuracy.
- Visualize training/validation accuracy and loss.
- Predict labels on a separate test dataset.
- Evaluate model performance with a pass threshold of 63%.

## Setup and Installation

This project requires Python 3.x and the following packages:

- tensorflow (version 2.x)
- numpy
- matplotlib
- wget (or use alternative file download)
- unzip (command-line tool or equivalent)

You can install the necessary Python packages using pip:

```bash
pip install tensorflow numpy matplotlib wget


# Health Insurance Cost Prediction

## Overview
This project is part of the FreeCodeCamp (FCC) course submission. It builds a deep learning model using TensorFlow and Keras to predict medical expenses based on various demographic and health-related factors. The dataset used comes from an insurance dataset containing features such as age, BMI, number of children, smoking status, and region.

## Dataset
The dataset is obtained from FreeCodeCamp and includes the following features:
- **age**: Age of the individual
- **sex**: Gender of the individual
- **bmi**: Body Mass Index (BMI)
- **children**: Number of children/dependents
- **smoker**: Whether the individual is a smoker or not
- **region**: Residential region
- **expenses**: Medical expenses (target variable)

## Project Workflow
1. **Data Preprocessing**:
   - One-hot encoding of categorical features.
   - Standard scaling of numerical features.
   - Log transformation of the target variable (`expenses`).
   
2. **Model Development**:
   - A feedforward neural network is implemented with three dense layers using ReLU activation.
   - Batch normalization and dropout layers are used for stability and overfitting prevention.
   - The model is compiled with the Adam optimizer and Mean Absolute Error (MAE) loss function.

3. **Training and Evaluation**:
   - Training is performed with early stopping and learning rate scheduling.
   - The model is validated on a test set, and predictions are compared with actual values.
   - MAE is computed on both the log-transformed and original scale.

## Model Performance
- The model aims to achieve a **Mean Absolute Error (MAE) < 3500** on the original scale of expenses.
- A scatter plot of **True vs. Predicted Expenses** is generated to visualize the performance.

## Results & Visualization
- The model outputs a histogram of predicted expenses.
- A scatter plot of actual vs. predicted values helps in understanding prediction accuracy.
- The final MAE is printed to determine if the challenge is passed.

## Potential Improvements
- **Feature Engineering**: Adding interaction terms or polynomial features.
- **Hyperparameter Tuning**: Adjusting learning rates, batch size, and number of neurons.
- **Alternative Models**: Exploring Random Forests, Gradient Boosting, or other deep learning architectures.

## Contributing
This project is part of the FreeCodeCamp (FCC) course submission. Contributions are welcome for learning purposes.

## License
This project is licensed under the MIT License.

---

### **Author**
- **Akshay Waghela**



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

## Conclusion
This project successfully implements a content-based book recommendation system using KNN and cosine similarity. It allows users to find similar books based on user ratings.

## References
- [FreeCodeCamp Book-Crossings Dataset](https://cdn.freecodecamp.org/project-data/books/book-crossings.zip)
- [Scikit-learn NearestNeighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Plotly Documentation](https://plotly.com/python/)

