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


