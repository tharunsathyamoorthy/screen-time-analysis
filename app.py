from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import difflib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
import math

app = Flask(__name__)
CORS(app)

POSSIBLE_COLUMNS = {
    "app_usage_time_min_day": 60,
    "app_usage_time_min_per_day": 60,
    "screen_time_min": 60,
    "screen_time_mins": 60,
    "minutes": 60,
    "minute": 60,
    "screen_time_sec": 3600,
    "seconds": 3600,
    "second": 3600,
    "screen_on_time_hours_day": 1,
    "screen_on_time_hours_per_day": 1,
    "screen_on_time": 1,
    "screen_time_hr": 1,
    "hours": 1,
    "hour": 1,
    "average_screen_time_hours": 1,
    "avg_daily_screen_time_hr": 1,
    "total_screentime": 1,
    "total_screen_time": 1,
    "daily_usage_hours": 1,
    "daily_screen_time_hours": 1,
}

VISION_CATEGORIES = [
    "Low Exposure - Healthy Visual Ergonomics",
    "Moderate Exposure - Normal Ocular Endurance",
    "High Exposure - Digital Eye Strain Risk"
]

def assign_age_group(age):
    if pd.isna(age):
        return "Unknown"
    group_start = (int(age) - 1) // 10 * 10 + 1
    group_end = group_start + 9
    return f"{group_start} - {group_end}"

def preprocess(df):
    normalized_cols = {
        col: col.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "_")
        for col in df.columns
    }
    col_found, factor = None, None
    for orig_col, norm_col in normalized_cols.items():
        if norm_col in POSSIBLE_COLUMNS:
            col_found = orig_col
            factor = POSSIBLE_COLUMNS[norm_col]
            break
    if not col_found:
        all_possible = list(POSSIBLE_COLUMNS.keys())
        close_matches = set()
        for norm_col in normalized_cols.values():
            close_matches.update(difflib.get_close_matches(norm_col, all_possible, n=2))
        return None, {
            "error": "No recognizable screen time column found in dataset.",
            "available_columns": list(normalized_cols.values()),
            "expected_columns": list(close_matches)
        }
    df["screen_time_hr"] = df[col_found] / factor
    return df, {"screen_time_column": col_found}

def generate_visualizations(df):
    # First: Vision Risk Distribution
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    status_counts = df['Vision_Status'].value_counts()
    colors = ['green', 'orange', 'red']
    bars = ax1.bar(status_counts.index, status_counts.values, color=colors[:len(status_counts)])
    ax1.set_title("Vision Risk Distribution")
    ax1.set_xlabel("Vision_Status")
    ax1.set_ylabel("Count")
    ax1.set_xticklabels(status_counts.index, rotation=45)

    static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)
    filename1 = f"vision_risk_dist_{uuid.uuid4().hex}.png"
    filepath1 = os.path.join(static_folder, filename1)
    plt.tight_layout()
    plt.savefig(filepath1)
    plt.close(fig1)

    # Second: Vision Risk by Age Group
    fig2, ax2 = plt.subplots(figsize=(7,6))
    if 'Age_Group' not in df.columns:
        age_group_map = df["Age"].apply(lambda age: "Unknown" if pd.isna(age) else (
            "18-25" if age <= 25 else "26-35" if age <= 35 else "36-45" if age <= 45 else "46-55" if age <= 55 else "56+"))
        df["Age_Group"] = age_group_map
    age_risk = pd.crosstab(df["Age_Group"], df["Vision_Status"])
    age_risk.plot(kind="bar", ax=ax2)
    ax2.set_title("Vision Risk by Age Group")
    ax2.set_xlabel("Age_Group")
    ax2.set_ylabel("count")
    plt.legend(title="Risk Category")
    plt.tight_layout()
    filename2 = f"vision_age_group_{uuid.uuid4().hex}.png"
    filepath2 = os.path.join(static_folder, filename2)
    plt.savefig(filepath2)
    plt.close(fig2)

    # Return both image URLs
    return [f"/static/{filename1}", f"/static/{filename2}"]

@app.route('/upload', methods=['POST'])
def upload_and_analyze():
    if 'file' not in request.files:
        return jsonify({"error": "CSV file is required"}), 400
    file = request.files['file']
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400
    df, info = preprocess(df)
    if df is None:
        return jsonify(info), 400

    # Advanced features & encoding if applicable
    if all(k in df.columns for k in ["Social_Media_Usage_Hours", "Gaming_App_Usage_Hours", "Productivity_App_Usage_Hours", 
                                     "Daily_Screen_Time_Hours", "Age", "Gender", "Location"]):
        df['Total_App_Usage'] = df['Social_Media_Usage_Hours'] + df['Gaming_App_Usage_Hours'] + df['Productivity_App_Usage_Hours']
        df['Screen_App_Ratio'] = df['Daily_Screen_Time_Hours'] / (df['Total_App_Usage'] + 0.001)
        df['Social_Media_Ratio'] = df['Social_Media_Usage_Hours'] / (df['Total_App_Usage'] + 0.001)
        df['Gaming_Ratio'] = df['Gaming_App_Usage_Hours'] / (df['Total_App_Usage'] + 0.001)
        df['Productivity_Ratio'] = df['Productivity_App_Usage_Hours'] / (df['Total_App_Usage'] + 0.001)
        df['App_Diversity'] = df[['Social_Media_Usage_Hours', 'Gaming_App_Usage_Hours', 'Productivity_App_Usage_Hours']].std(axis=1)
        df['Usage_Efficiency'] = df['Total_App_Usage'] / (df['Daily_Screen_Time_Hours'] + 0.001)

        def assign_grp(age):
            if age <= 25:
                return "18-25"
            elif age <= 35:
                return "26-35"
            elif age <= 45:
                return "36-45"
            elif age <= 55:
                return "46-55"
            else:
                return "56+"
        df["Age_Group"] = df["Age"].apply(assign_grp)

        le_gender = LabelEncoder()
        df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
        le_location = LabelEncoder()
        df['Location_Encoded'] = le_location.fit_transform(df['Location'])

        q1 = df["Daily_Screen_Time_Hours"].quantile(0.33)
        q2 = df["Daily_Screen_Time_Hours"].quantile(0.66)

        def predict_vision_risk(t):
            if t <= q1:
                return "Low Exposure - Healthy Visual Ergonomics"
            elif t <= q2:
                return "Moderate Exposure - Normal Ocular Endurance"
            else:
                return "High Exposure - Digital Eye Strain Risk"

        df["Vision_Status"] = df["Daily_Screen_Time_Hours"].apply(predict_vision_risk)
    else:
        df["Vision_Status"] = df["screen_time_hr"].apply(
            lambda t: "High Exposure - Digital Eye Strain Risk" if t > df["screen_time_hr"].mean()
            else ("Low Exposure - Healthy Visual Ergonomics" if t < df["screen_time_hr"].mean() else "Moderate Exposure - Normal Ocular Endurance")
        )

    # Generate charts and get urls
    visualizations = generate_visualizations(df)

    # Prepare response
    processed_data_sample = df.head(10).to_dict(orient="records")
    avg_screen_time = df["screen_time_hr"].mean()
    threshold = avg_screen_time
    exceed_pct = (df["screen_time_hr"] > threshold).mean() * 100
    health_trends = df["Vision_Status"].value_counts().reindex(VISION_CATEGORIES, fill_value=0).to_dict()

    return jsonify({
        "average_screen_time_hours": avg_screen_time,
        "recommendation_hours": threshold,
        "percentage_exceeding_recommendation": exceed_pct,
        "health_impact_trends": health_trends,
        "screen_time_column": info["screen_time_column"],
        "processed_data_sample": processed_data_sample,
        "charts": {
            "visualizations": visualizations
        }
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
