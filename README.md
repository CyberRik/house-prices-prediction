
# House Price Prediction App

This repository contains a **House Price Prediction** model built using **machine learning** techniques. The model is based on a dataset of house prices and aims to predict the price of a house based on various features such as quality, garage space, and more.

The web app is built with **Streamlit** and utilizes a **pre-trained pipeline model** to predict house prices in real-time.

## Overview

The house price prediction system allows users to input various features of a house, such as:
- Overall quality
- Garage space
- Living area
- And other factors that influence house pricing.

The app then predicts the house price based on the provided inputs, compares it with the training set prices (min, average, and max), and visualizes the comparison. It also displays feature importance, showing which features most influence the predicted house price.

## Technologies Used

- **Python**
- **Streamlit**: For creating the interactive web interface.
- **Scikit-learn**: For machine learning model training and preprocessing.
- **XGBoost**: For the house price prediction model.
- **Matplotlib & Seaborn**: For visualizing results and feature importance.

## How to Run the App Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   ```

2. Navigate into the project directory:
   ```bash
   cd house-price-prediction
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and go to `http://localhost:8501` to start using the house price predictor.

## Model Details

The model used in this application is a **pipeline** created using **XGBoost** and **Scikit-learn**, trained on the provided house price dataset (`train.csv`). It includes preprocessing steps for both numerical and categorical features. The model predicts house prices based on the selected features, and the prediction results are visualized alongside comparison data (min, average, and max prices).

### Key Features:
- **Input fields**: Users can input various house features using sliders and dropdown menus.
- **Price Prediction**: Upon clicking the "Predict House Price" button, the predicted price is shown.
- **Feature Importance**: The app displays a bar plot of the most important features affecting the house price prediction.

## Files in the Repository

- **`app.py`**: Main application file that runs the Streamlit app and handles predictions.
- **`house_price_pipeline.pkl`**: Pre-trained model pipeline for predicting house prices.
- **`train.csv`**: Training dataset containing house features and their respective prices.
- **`test.csv`**: Test dataset for validation (optional).
- **`requirements.txt`**: A list of required Python packages.
- **`README.md`**: This file.

## Example Usage

After running the app, you can interact with the input sliders and dropdowns to provide the features of a house you're interested in, and the model will predict the price for you.

![Predicted House Price](images/price_prediction.png)

## Feature Importance

The app also provides a visualization of the **top 10 features** that contribute the most to the price prediction. This is a helpful tool for understanding which features influence house prices the most.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to fork this repository and submit pull requests for improvements or fixes. If you encounter any issues, open an issue in the repository.

---

Made with ❤️ by [CyberRik](https://github.com/CyberRik)
