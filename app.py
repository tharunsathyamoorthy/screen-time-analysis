from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import difflib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid

app = Flask(__name__)
CORS(app)

# Global DataFrame to reuse for /analyze_age
uploaded_df = None

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
    static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)
    img_urls = []

    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    if 'Age_Group' not in df.columns:
        age_group_map = df["Age"].apply(lambda age: "Unknown" if pd.isna(age) else (
            "18-25" if age <= 25 else "26-35" if age <= 35 else "36-45" if age <= 45 else "46-55" if age <= 55 else "56+"))
        df["Age_Group"] = age_group_map
    age_risk = pd.crosstab(df["Age_Group"], df["Vision_Status"])
    line_colors = ['#0072BD', '#ED7D31', '#228B22']

    for idx, status in enumerate(age_risk.columns):
        ax2.plot(
            age_risk.index, age_risk[status],
            marker='o', linewidth=3, label=status, color=line_colors[idx % len(line_colors)]
        )
        for i, v in enumerate(age_risk[status]):
            ax2.annotate(
                str(v), (i, v), textcoords="offset points", xytext=(0, 8),
                ha='center', fontsize=13, color=line_colors[idx % len(line_colors)], fontweight='bold'
            )
    ax2.set_title("Vision Risk by Age Group", fontsize=18, fontweight='bold', pad=15)
    ax2.set_xlabel("Age Group", fontsize=15)
    ax2.set_ylabel("Count", fontsize=15)
    ax2.legend(title="Risk Category", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    fig2.tight_layout()
    file2 = f"vision_age_group_{uuid.uuid4().hex}.png"
    filepath2 = os.path.join(static_folder, file2)
    plt.savefig(filepath2, dpi=180, bbox_inches='tight')
    plt.close(fig2)
    img_urls.append(f"/static/{file2}")

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    status_counts = df['Vision_Status'].value_counts()
    ax1.plot(
        status_counts.index, status_counts.values, marker='o',
        color='#1f77b4', linewidth=3
    )
    for i, value in enumerate(status_counts.values):
        ax1.annotate(
            str(value), (i, value), textcoords="offset points", xytext=(0, 8),
            ha='center', fontsize=14, color='#1f77b4', fontweight='bold'
        )
    ax1.set_title("Vision Risk Distribution", fontsize=18, fontweight='bold', pad=15)
    ax1.set_xlabel("Vision Status", fontsize=15)
    ax1.set_ylabel("Count", fontsize=15)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xticks(range(len(status_counts.index)))
    ax1.set_xticklabels(status_counts.index, rotation=20, ha='right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    fig1.tight_layout()
    file1 = f"vision_risk_dist_{uuid.uuid4().hex}.png"
    filepath1 = os.path.join(static_folder, file1)
    plt.savefig(filepath1, dpi=180, bbox_inches='tight')
    plt.close(fig1)
    img_urls.append(f"/static/{file1}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    sns.histplot(df["Daily_Screen_Time_Hours"], bins=20, kde=True, color="blue", ax=axes[0, 0])
    axes[0, 0].axvline(df["Daily_Screen_Time_Hours"].mean(), color="red", linestyle="--", label="Mean")
    axes[0, 0].set_title("Distribution of Daily Screen Time (Hours)")
    axes[0, 0].set_xlabel("Screen Time (hours)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()

    usage_cols = ["Social_Media_Usage_Hours", "Productivity_App_Usage_Hours", "Gaming_App_Usage_Hours"]
    avg_usage = df[usage_cols].mean().reset_index()
    avg_usage.columns = ["App_Type", "Average_Hours"]
    axes[0, 1].plot(avg_usage["App_Type"], avg_usage["Average_Hours"], marker='o', color='purple', linewidth=3)
    for i, v in enumerate(avg_usage["Average_Hours"]):
        axes[0, 1].text(i, v, f"{v:.2f}", color='purple', fontweight='bold', ha='center', va='bottom', fontsize=13)
    axes[0, 1].set_title("Average Time Spent by App Category")
    axes[0, 1].set_ylabel("Average Hours/Day")
    axes[0, 1].tick_params(axis='x', rotation=45)

    gender_group = df.groupby("Gender")["Daily_Screen_Time_Hours"].mean().reset_index()
    axes[1, 0].plot(gender_group["Gender"], gender_group["Daily_Screen_Time_Hours"], marker='o', color='teal', linewidth=3)
    for i, v in enumerate(gender_group["Daily_Screen_Time_Hours"]):
        axes[1, 0].text(i, v, f"{v:.2f}", color='teal', fontweight='bold', ha='center', va='bottom', fontsize=13)
    axes[1, 0].set_title("Screen Time Distribution by Gender")
    axes[1, 0].set_xlabel("Gender")
    axes[1, 0].set_ylabel("Average Daily Screen Time (Hours)")

    age_group = df.groupby("Age_Group")["Daily_Screen_Time_Hours"].mean().reset_index()
    axes[1, 1].plot(age_group["Age_Group"], age_group["Daily_Screen_Time_Hours"], marker='o', color='orange', linewidth=3)
    for i, v in enumerate(age_group["Daily_Screen_Time_Hours"]):
        axes[1, 1].text(i, v, f"{v:.2f}", color='orange', fontweight='bold', ha='center', va='bottom', fontsize=13)
    axes[1, 1].set_title("Screen Time Distribution by Age Group")
    axes[1, 1].set_xlabel("Age_Group")
    axes[1, 1].set_ylabel("Average Daily Screen Time (Hours)")

    plt.tight_layout()
    filename = f"visualizations_{uuid.uuid4().hex}.png"
    filepath = os.path.join(static_folder, filename)
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()
    img_urls.append(f"/static/{filename}")

    return img_urls

@app.route('/upload', methods=['POST'])
def upload_and_analyze():
    global uploaded_df
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
                return "Low Exposure"
            elif t <= q2:
                return "Moderate Exposure"
            else:
                return "High Exposure"

        df["Vision_Status"] = df["Daily_Screen_Time_Hours"].apply(predict_vision_risk)
    else:
        df["Vision_Status"] = df["screen_time_hr"].apply(
            lambda t: "High Exposure - Digital Eye Strain Risk" if t > df["screen_time_hr"].mean()
            else ("Low Exposure - Healthy Visual Ergonomics" if t < df["screen_time_hr"].mean() else "Moderate Exposure - Normal Ocular Endurance")
        )

    features = []
    if all(col in df.columns for col in ['Social_Media_Usage_Hours', 'Gaming_App_Usage_Hours', 'Productivity_App_Usage_Hours', 'Daily_Screen_Time_Hours', 'Gender_Encoded', 'Location_Encoded']):
        features = ['Social_Media_Usage_Hours', 'Gaming_App_Usage_Hours', 'Productivity_App_Usage_Hours', 'Daily_Screen_Time_Hours', 'Gender_Encoded', 'Location_Encoded']
    else:
        features = ['screen_time_hr']

    X = df[features].fillna(0).values
    label_mapping = {name: idx for idx, name in enumerate(VISION_CATEGORIES)}
    y = df["Vision_Status"].map(label_mapping).fillna(0).astype(int)
    y_cat = to_categorical(y, num_classes=len(VISION_CATEGORIES))

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(32, activation='relu'),
        Dense(len(VISION_CATEGORIES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)

    # Model accuracy evaluation and print to terminal
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Model Test Accuracy: {accuracy:.4f}")

    predictions_prob = model.predict(X)
    predictions = predictions_prob.argmax(axis=1)
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    df['Predicted_Vision_Status'] = [inv_label_mapping.get(p, VISION_CATEGORIES[0]) for p in predictions]

    uploaded_df = df.copy()
    visualization_urls = generate_visualizations(df)

    HIGH_VISION_RISK_PRECAUTIONS = [
        "Limit continuous screen time to no more than 30–40 minutes at a stretch.",
        "Follow the 20-20-20 rule: take a 20-second break every 20 minutes by looking at something 20 feet away.",
        "Use blue light filters or night modes on all devices, especially after sunset.",
        "Ensure proper ambient lighting to reduce glare and avoid working in dark rooms.",
        "Place screens at eye level and about 18-24 inches from your eyes.",
        "Blink frequently to minimize dryness; consider lubricating drops if needed.",
        "Schedule regular eye exams and seek advice if persistent discomfort occurs.",
        "Keep device screens clean to avoid straining over smudges or dirt.",
        "Practice comprehensive eye exercises (focus shifting, rolling, palming).",
        "Limit multitasking across multiple screens/devices at the same time.",
        "Maintain ergonomic posture, supporting your back, shoulders, and neck.",
        "Consider using screen time management apps to monitor and reduce your digital exposure.",
    ]

    MODERATE_VISION_RISK_PRECAUTIONS = [
        "Set daily limits for total recreational screen time, aiming for under 2 hours outside work/study.",
        "Integrate frequent, short breaks to avoid prolonged sessions.",
        "Use blue light reduction settings in the evening and increase ambient light during the day.",
        "Increase physical activity and balance screen work with outdoor breaks.",
        "Monitor for early symptoms: dryness, headaches, blurry vision.",
        "Prioritize larger screens for reading or extended work sessions over small mobile devices.",
        "Avoid staring at small text or poorly contrasted screens for long periods.",
        "Check device ergonomics—screen tilt, height, and brightness should minimize squinting.",
        "Plan device-free family or personal time blocks daily.",
        "Encourage children/adolescents to follow healthy device habits with gentle supervision.",
        "Wear prescription eyewear if indicated and consult an optometrist for special screen-use lenses if discomfort arises.",
    ]

    LOW_VISION_RISK_PRECAUTIONS = [
        "Continue healthy visual habits and regular blinking.",
        "Include screen-free activities in your daily routine.",
        "Maintain good lighting at your workspace.",
        "Stay hydrated for optimal eye health.",
        "Monitor for new screen-related symptoms, even if rare.",
        "Adjust brightness to natural room light; avoid overly bright/dim screens.",
        "Periodically check your device usage to ensure minimal risk.",
        "Dedicate time for outdoor eyesight exercise (natural focusing).",
        "Avoid using screens while commuting or in moving vehicles.",
        "Prioritize sleep routines by finishing screen work well before bedtime.",
    ]

    vision_precautions = {
        "High Exposure": HIGH_VISION_RISK_PRECAUTIONS,
        "Moderate Exposure": MODERATE_VISION_RISK_PRECAUTIONS,
        "Low Exposure": LOW_VISION_RISK_PRECAUTIONS,
    }

    processed_data_sample = df.head(10).to_dict(orient="records")
    avg_screen_time = df["screen_time_hr"].mean()
    threshold = 6  # Set threshold to 6 hours per day
    exceed_pct = (df["screen_time_hr"] > threshold).mean() * 100
    exposure_map = {
        "Low Exposure": "Low Exposure - Healthy Visual Ergonomics",
        "Moderate Exposure": "Moderate Exposure - Normal Ocular Endurance",
        "High Exposure": "High Exposure - Digital Eye Strain Risk"
    }
    vision_status_long = df["Vision_Status"].map(exposure_map)
    health_trends = vision_status_long.value_counts().reindex(VISION_CATEGORIES, fill_value=0).to_dict()

    return jsonify({
        "average_screen_time_hours": avg_screen_time,
        "recommendation_hours": threshold,
        "percentage_exceeding_recommendation": exceed_pct,
        "health_impact_trends": health_trends,
        "screen_time_column": info["screen_time_column"],
        "processed_data_sample": processed_data_sample,
        "charts": {
            "visualizations": visualization_urls
        },
        "vision_precautions": vision_precautions,
        "model_test_accuracy": float(accuracy)
    })

@app.route('/analyze_age', methods=['POST'])
def analyze_age():
    global uploaded_df
    if uploaded_df is None:
        return jsonify({"error": "No dataset uploaded yet. Please upload a file first."}), 400

    data = request.get_json()
    if not data or "age" not in data:
        return jsonify({"error": "Age value required in JSON format (e.g. {'age': 25})"}), 400

    try:
        age = int(data["age"])
    except Exception:
        return jsonify({"error": "Invalid age value."}), 400

    if age <= 25:
        group = "18-25"
    elif age <= 35:
        group = "26-35"
    elif age <= 45:
        group = "36-45"
    elif age <= 55:
        group = "46-55"
    else:
        group = "56+"

    if 'Age_Group' not in uploaded_df.columns:
        uploaded_df['Age_Group'] = uploaded_df['Age'].apply(assign_age_group)

    group_df = uploaded_df[uploaded_df['Age_Group'] == group]

    if group_df.empty:
        return jsonify({"message": f"No records found for age group {group}."}), 404

    # --- Model training and accuracy calculation for age group ---
    # Use only group_df for training and evaluation
    features = []
    if all(col in group_df.columns for col in ['Social_Media_Usage_Hours', 'Gaming_App_Usage_Hours', 'Productivity_App_Usage_Hours', 'Daily_Screen_Time_Hours', 'Gender_Encoded', 'Location_Encoded']):
        features = ['Social_Media_Usage_Hours', 'Gaming_App_Usage_Hours', 'Productivity_App_Usage_Hours', 'Daily_Screen_Time_Hours', 'Gender_Encoded', 'Location_Encoded']
    else:
        features = ['screen_time_hr']

    X = group_df[features].fillna(0).values
    label_mapping = {name: idx for idx, name in enumerate(VISION_CATEGORIES)}
    y = group_df["Vision_Status"].map(label_mapping).fillna(0).astype(int)
    y_cat = to_categorical(y, num_classes=len(VISION_CATEGORIES))

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(32, activation='relu'),
        Dense(len(VISION_CATEGORIES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)

    # Model accuracy evaluation and print to terminal for age group
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Model Test Accuracy for Age Group {group}: {accuracy:.4f}")

    static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)

    import matplotlib
    matplotlib.use('Agg')

    fig, ax = plt.subplots(figsize=(7, 5))
    counts = group_df["Vision_Status"].value_counts()
    ax.plot(counts.index, counts.values, marker='o', color='#1f77b4', linewidth=3)
    for i, value in enumerate(counts.values):
        ax.annotate(
            str(value), (i, value), textcoords="offset points", xytext=(0, 8),
            ha='center', fontsize=12, color='#1f77b4', fontweight='bold'
        )
    ax.set_title(f"Vision Status Distribution for Age Group {group}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Vision Status")
    ax.set_ylabel("Count")
    plt.xticks(rotation=20)
    plt.tight_layout()

    file_name = f"age_group_{group}_{uuid.uuid4().hex}.png"
    file_path = os.path.join(static_folder, file_name)
    plt.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close()

    chart_url = f"/static/{file_name}"

    cols = [c for c in ['Name', 'Age', 'Gender', 'Vision_Status', 'screen_time_hr'] if c in group_df.columns]
    records = group_df[cols].head(20).to_dict(orient='records')

    vision_status_summary = counts.to_dict()

    HIGH_VISION_RISK_PRECAUTIONS = [
        "Limit continuous screen time to no more than 30–40 minutes at a stretch.",
        "Follow the 20-20-20 rule: take a 20-second break every 20 minutes by looking at something 20 feet away.",
        "Use blue light filters or night modes on all devices, especially after sunset.",
        "Ensure proper ambient lighting to reduce glare and avoid working in dark rooms.",
        "Place screens at eye level and about 18-24 inches from your eyes.",
        "Blink frequently to minimize dryness; consider lubricating drops if needed.",
        "Schedule regular eye exams and seek advice if persistent discomfort occurs.",
        "Keep device screens clean to avoid straining over smudges or dirt.",
        "Practice comprehensive eye exercises (focus shifting, rolling, palming).",
        "Limit multitasking across multiple screens/devices at the same time.",
        "Maintain ergonomic posture, supporting your back, shoulders, and neck.",
        "Consider using screen time management apps to monitor and reduce your digital exposure.",
    ]

    MODERATE_VISION_RISK_PRECAUTIONS = [
        "Set daily limits for total recreational screen time, aiming for under 2 hours outside work/study.",
        "Integrate frequent, short breaks to avoid prolonged sessions.",
        "Use blue light reduction settings in the evening and increase ambient light during the day.",
        "Increase physical activity and balance screen work with outdoor breaks.",
        "Monitor for early symptoms: dryness, headaches, blurry vision.",
        "Prioritize larger screens for reading or extended work sessions over small mobile devices.",
        "Avoid staring at small text or poorly contrasted screens for long periods.",
        "Check device ergonomics—screen tilt, height, and brightness should minimize squinting.",
        "Plan device-free family or personal time blocks daily.",
        "Encourage children/adolescents to follow healthy device habits with gentle supervision.",
        "Wear prescription eyewear if indicated and consult an optometrist for special screen-use lenses if discomfort arises.",
    ]

    LOW_VISION_RISK_PRECAUTIONS = [
        "Continue healthy visual habits and regular blinking.",
        "Include screen-free activities in your daily routine.",
        "Maintain good lighting at your workspace.",
        "Stay hydrated for optimal eye health.",
        "Monitor for new screen-related symptoms, even if rare.",
        "Adjust brightness to natural room light; avoid overly bright/dim screens.",
        "Periodically check your device usage to ensure minimal risk.",
        "Dedicate time for outdoor eyesight exercise (natural focusing).",
        "Avoid using screens while commuting or in moving vehicles.",
        "Prioritize sleep routines by finishing screen work well before bedtime.",
    ]

    vision_precautions = {
        "High Exposure": HIGH_VISION_RISK_PRECAUTIONS,
        "Moderate Exposure": MODERATE_VISION_RISK_PRECAUTIONS,
        "Low Exposure": LOW_VISION_RISK_PRECAUTIONS,
    }

    return jsonify({
        "age_group": group,
        "total_records": int(len(group_df)),
        "vision_status_summary": vision_status_summary,
        "chart_url": chart_url,
        "records": records,
        "vision_precautions": vision_precautions  # include existing precautions for frontend use
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
