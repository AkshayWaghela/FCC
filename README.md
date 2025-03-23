# FCC


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

## Installation & Setup
To run this project, follow these steps:

### **1. Clone the repository**
```bash
git clone https://github.com/your-username/health-insurance-prediction.git
cd health-insurance-prediction
```

### **2. Install dependencies**
Ensure you have Python 3.x installed. Then, install the required libraries:
```bash
pip install -r requirements.txt
```

### **3. Run the script**
Execute the following command:
```bash
python insurance_cost_prediction.py
```

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

