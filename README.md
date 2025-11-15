# SmartCart-ML-Project
## 📌 Project Overview
SmartCart is a Machine Learning–based shopping recommendation system designed to suggest 
relevant products to users based on the last item added to their cart. The project demonstrates 
an end-to-end ML workflow including data preprocessing, model training in Google Colab 
using an advanced dataset, exporting the trained model as joblib files, and deploying it inside 
a Streamlit web application. This aligns with the requirements of the Computational Intelligence 
course by showcasing a complete ML lifecycle integrated into a real-time interface.

The system predicts the approximate price category of a product using a Random Forest 
Regressor and recommends another product from the same price range. This creates a simple 
yet effective recommendation mechanism similar to basic e-commerce platforms.

## 📁 Project Files Description

### 1. Application Files (Streamlit App)
- **app.py**  
  Main Streamlit application file. Handles product display, cart system, 
  loading the ML model, and generating recommendations.

### 2. Machine Learning Model Files
- **best_model.joblib**  
  Trained Random Forest Regressor used for predicting price categories.
- **encoder.joblib**  
  Saved LabelEncoder used to convert product names into numerical values.
- **scaler.joblib**  
  Saved StandardScaler used to normalize encoded values before prediction.
- **train_new_model.py**  
  Python script used to train the ML model and save the joblib files.

### 3. Datasets
- **advanced_smartcart_dataset.csv**  
  Preprocessed and encoded dataset used for ML model training in Google Colab.
- **processed_smartcart.csv**  
  A version of the dataset after preprocessing (cleaning, encoding, scaling).
- **products.csv**  
  Simple dataset used in the Streamlit UI containing product names, prices, and images.

### 4. Jupyter Notebook
- **advancedshoppingcart.ipynb**  
  Google Colab notebook used for data preprocessing, model training, evaluation, 
  and generating the joblib files.

### 5. Report Files
- **SmartCart_ProjectReport.pdf**  
  Final academic report describing the methodology, model, datasets, results, and conclusions.

