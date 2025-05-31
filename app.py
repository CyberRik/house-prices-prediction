import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load('house_price_pipeline.pkl')
train_df = pd.read_csv('train.csv')

#preprocessing

numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

default_values = {}
for col in numerical_cols:
    default_values[col] = train_df[col].mean()

for col in categorical_cols:
    default_values[col] = train_df[col].mode()[0]

#top features from importance analysis

top_features = [
    'OverallQual', 'GarageCars', 'KitchenQual', 'BsmtQual', 'RoofStyle',
    'CentralAir', 'MSZoning', 'GrLivArea', 'LotShape', 'GarageFinish'
]

numerical_features = ['OverallQual', 'GarageCars', 'GrLivArea']
categorical_features = list(set(top_features) - set(numerical_features))

categorical_feature_values = {}
for feature in categorical_features:
    if feature in train_df.columns:
        categorical_feature_values[feature] = train_df[feature].dropna().unique().tolist()

st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction")
st.write("Enter the details below to predict the house price:")

#input form

user_input = {}
for feature in numerical_features:
    user_input[feature] = st.slider(f"{feature}", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

for feature in categorical_features:
    if feature in categorical_feature_values:
        user_input[feature] = st.selectbox(f"{feature}", categorical_feature_values[feature])
    else:
        user_input[feature] = st.selectbox(f"{feature}", ['None'])

st.write("### User Inputs")
st.write(pd.DataFrame([user_input]))

input_df = pd.DataFrame(columns=model.named_steps['preprocessor'].feature_names_in_)
for col in input_df.columns:
    if col in user_input:
        input_df.at[0, col] = user_input[col]
    else:
        input_df.at[0, col] = default_values.get(col, np.nan)

#Button to predict house price

if st.button("Predict House Price", key="predict_button"):
    prediction = model.predict(input_df)[0]
    st.success(f"🏡 **Predicted House Price**: ${prediction:,.2f}")

    min_price = train_df['SalePrice'].min()
    max_price = train_df['SalePrice'].max()
    mean_price = train_df['SalePrice'].mean()

    comparison_df = pd.DataFrame({
        'Price Type': ['Min Price', 'Average Price', 'Predicted Price', 'Max Price'],
        'Price Value': [min_price, mean_price, prediction, max_price]
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Price Value', y='Price Type', data=comparison_df, ax=ax, palette="coolwarm")

    ax.set_title("Predicted House Price vs. Training Set Prices", fontsize=16, fontweight='bold')
    ax.set_xlabel("Price in $", fontsize=12)
    ax.set_ylabel("", fontsize=12)

    for i in range(len(comparison_df)):
        ax.text(comparison_df['Price Value'].iloc[i] + 5000, i, 
                f'{comparison_df["Price Value"].iloc[i]:,.0f}', 
                color='black', ha="left", va="center", fontsize=10)

    st.pyplot(fig)

# Feature Importance Analysis

st.write("### Feature Importance")
importances = model.named_steps['model'].feature_importances_
feature_names = model.named_steps['preprocessor'].get_feature_names_out()

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df['Feature'] = importance_df['Feature'].str.replace('num__', '').str.replace('cat__', '')

top_10_importance = importance_df.head(10)
fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(x="Importance", y="Feature", data=top_10_importance, ax=ax, palette="coolwarm")
ax.set_title("Top 10 Important Features for House Price Prediction", fontsize=16, fontweight='bold')
ax.set_xlabel("Importance", fontsize=12)
ax.set_ylabel("Features", fontsize=12)
ax.tick_params(axis='both', labelsize=10)
ax.grid(False)

for i in range(len(top_10_importance)):
    ax.text(top_10_importance['Importance'].iloc[i] + 0.01, i, 
            f'{top_10_importance["Importance"].iloc[i]:.4f}', 
            color='black', ha="left", va="center", fontsize=10)

st.pyplot(fig)

st.markdown(
    """
    <div style="display: flex; margin-top: 50px;">
        <a href="https://github.com/CyberRik/house-prices-prediction" target="_blank">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" width="50"/>
        </a>
    </div>
    """, unsafe_allow_html=True
)
st.write("Made with ❤️ by [CyberRik]")
