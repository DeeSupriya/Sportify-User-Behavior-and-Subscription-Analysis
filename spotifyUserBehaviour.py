# =================================================
#           Spotify User Behaviuor Analysis
# =================================================
# Author: Deekhsa

#import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load Datasets
df = pd.read_csv("spotify_user_behavior_realistic_50000_rows.csv")

print("First 5 rows:\n", df.head())
print("\nDatasets Info:\n", df.info())

# =========================================
#           Data Cleaning
# =========================================

# Date Conversion
df['signup_date'] = pd.to_datetime(df["signup_date"], dayfirst=True)
df['signup_year']= df['signup_date'].dt.year
df['signup_month'] = df['signup_date'].dt.month

df = df.drop('signup_date', axis=1)


# Converting yes/no to 1/0, becoz macine doesn't undersatnd text data
binary_cols = ['ad_interaction', 'ad_conversion_to_subscription']
for col in binary_cols:
    df[col] = df[col].map({'Yes':1, 'No':0})

print("\nClass Distribution:\n")
print(df['ad_conversion_to_subscription'].value_counts())

# Check if any data is missing(empty/ NaN)
print("\nMissing Values:\n", df.isnull().sum())


# ==========================================
#           Feature Engineering
# ==========================================

# Engagement Score
df['engagement_score'] = df['avg_listening_hours_per_week'] / (df['avg_skips_per_day']+1) # for mathematical error


# ===========================================
#          EDA(Exploratory Data Analysis)
# ===========================================

# Setting Design of Graphs
sns.set(style="whitegrid")

# Distribution(Subscription Type)
plt.figure()
sns.countplot(data=df, x='subscription_type')
plt.title("Subscription Type Distribution")
plt.xticks(rotation=30)
plt.savefig("Subscription type distribution.jpeg")
plt.show()

# Age vs Listening Hours
plt.figure()
sns.scatterplot(data=df, x='age', y='avg_listening_hours_per_week')
plt.title("Age vs Listening Hours")
plt.savefig("age_vs_listening_hours.jpeg")
plt.show()

# Listening Hours by subscription
plt.figure()
sns.barplot(data=df, x='subscription_type', y='avg_listening_hours_per_week')
plt.title("Listening Hours by Subscription")
plt.xticks(rotation=30)
plt.savefig("Listening_hours_by_subscription.jpeg")
plt.show()

# Ad_intection vs Ad_conversion_to_subscription
plt.figure()
sns.countplot(data=df, x='ad_interaction', hue='ad_conversion_to_subscription')
plt.title("Ad Intersction vs Conversion")
plt.savefig("Ad_interaction_vs_conversion.jpeg")
plt.show()

# Favorite Genre Distribution
plt.figure()
sns.countplot(data=df, x='favorite_genre')
plt.title("Favorite Genre Distribution")
plt.xticks(rotation=30)
plt.savefig("Favorit_Genre_Distribution.jpeg")
plt.show()

# Devise usage
plt.figure()
sns.countplot(data=df, x='primary_device')
plt.title("Device Usage")
plt.xticks(rotation=30)
plt.savefig("Device_usage.jpeg")
plt.show()

# Skips vs Listening Hours
plt.figure()
sns.scatterplot(data=df, x='avg_skips_per_day', y='avg_listening_hours_per_week')
plt.title("Skips vs Listening Hours")
plt.savefig("skip_vs_listening_hours.jpeg")
plt.show()


# =========================================
#               ML Model
# =========================================
 
# Encoding categorical variables
le = LabelEncoder()
categorical_cols= df.select_dtypes(include=['object', 'string']).columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Define Features & Targets
X = df.drop('ad_conversion_to_subscription', axis=1)
y = df['ad_conversion_to_subscription']

# Train-Test Spilt
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================
#               Model Evaluation
# =============================================

# Model and Prediction
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# ==================================================
#           Feature Importance
# ==================================================

feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values().plot(kind='barh', title='Feature Importance')
plt.savefig("Feature_Importance.jpeg")
plt.show()


# ====================================================
#          Key Insights
# ====================================================

print("\nKey Insights:")
print("- Higher listening hours indicate higher engagement.")
print("- High skip rates suggest low satisfaction.")
print("Ad interaction increases chances of conversion.")
print("- Mobile devices are most commonly used.")
print("- Features like Daily Mix and AI DJ drive engagement.")