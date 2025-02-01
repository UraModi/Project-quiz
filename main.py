import json
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Define the correct data folder path
DATA_DIR = r"D:\Project\Project Quiz\data"

# File paths
CURRENT_QUIZ_FILE = os.path.join(DATA_DIR, "current_quiz.json")
HISTORICAL_QUIZ_FILE = os.path.join(DATA_DIR, "historical_quiz.json")

def load_json(filename):
    """Load JSON data from a file."""
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"Error: File not found: {filename}")
        return None

def process_current_quiz(data):
    """Convert current quiz data to a Pandas DataFrame."""
    if not data:
        print("No data to process for current quiz.")
        return None, None

    quiz_info = {
        "quiz_id": data.get("quiz_id"),
        "score": data.get("score"),
        "accuracy": data.get("accuracy"),
        "total_questions": data.get("total_questions"),
        "correct_answers": data.get("correct_answers"),
        "incorrect_answers": data.get("incorrect_answers"),
        "duration": data.get("duration"),
    }

    responses = [{"question_id": qid, "selected_option": option} for qid, option in data.get("response_map", {}).items()]

    df_quiz = pd.DataFrame(responses)
    df_metadata = pd.DataFrame([quiz_info])

    return df_quiz, df_metadata

def process_historical_quiz(data):
    """Convert historical quiz data to a Pandas DataFrame."""
    if not data:
        print("No data to process for historical quiz.")
        return None

    records = []
    for quiz in data:
        records.append({
            "quiz_id": quiz["quiz_id"],
            "score": quiz["score"],
            "accuracy": quiz["accuracy"],
            "total_questions": quiz["total_questions"],
            "correct_answers": quiz["correct_answers"],
            "incorrect_answers": quiz["incorrect_answers"],
            "duration": quiz["duration"],
            "response_map": quiz["response_map"]
        })

    df = pd.DataFrame(records)
    return df

# Load data from JSON files
current_quiz_data = load_json(CURRENT_QUIZ_FILE)
historical_quiz_data = load_json(HISTORICAL_QUIZ_FILE)

# Process Data
df_current_quiz, df_current_metadata = (process_current_quiz(current_quiz_data) if current_quiz_data else (None, None))
df_historical_quiz = process_historical_quiz(historical_quiz_data) if historical_quiz_data else None

# Advanced Analysis & Insights
def analyze_performance(df_current_quiz, df_current_metadata, df_historical_quiz):
    if df_current_quiz is not None:
        accuracy = df_current_metadata["accuracy"].iloc[0]
        correct_answers = df_current_metadata["correct_answers"].iloc[0]
        incorrect_answers = df_current_metadata["incorrect_answers"].iloc[0]

        print(f"Current Quiz Performance:")
        print(f"Accuracy: {accuracy}%")
        print(f"Correct Answers: {correct_answers} / Incorrect Answers: {incorrect_answers}")

        # Weak Areas: Questions answered incorrectly
        incorrect_responses = df_current_quiz[df_current_quiz['selected_option'] != 1]
        print(f"Topics to Focus on (Incorrect Responses):")
        print(incorrect_responses)

        # Historical Performance Trends
        if df_historical_quiz is not None:
            print("\nHistorical Performance Trends:")
            historical_scores = df_historical_quiz['score']
            historical_accuracy = df_historical_quiz['accuracy'].apply(lambda x: float(x.replace("%", "")) / 100)

            print(f"Average Score across last 5 quizzes: {historical_scores.mean()}")
            print(f"Average Accuracy across last 5 quizzes: {historical_accuracy.mean() * 100:.2f}%")

            # Plot Score Progression and Trends
            plt.figure(figsize=(10, 6))
            historical_scores.plot(kind='line', title="Score Progression", xlabel="Quiz Number", ylabel="Score")
            plt.show()

            # Visualizing difficulty-based accuracy
            difficulty_accuracy = df_historical_quiz.groupby("difficulty")["accuracy"].mean()
            plt.figure(figsize=(10, 6))
            sns.barplot(x=difficulty_accuracy.index, y=difficulty_accuracy.values)
            plt.title("Accuracy by Difficulty Level")
            plt.ylabel("Accuracy (%)")
            plt.show()

        # Provide Actionable Recommendations
        print("\nRecommendations:")
        if incorrect_responses.empty:
            print("Great job! Keep up the good performance!")
        else:
            print("Focus on the topics/questions you struggled with. Review the incorrect answers.")
            print("Target practicing more questions in these areas with similar difficulty.")

        # NEET Rank Prediction based on Machine Learning Model
        if df_historical_quiz is not None:
            recent_scores = df_historical_quiz[['score', 'accuracy', 'total_questions']]
            recent_scores['accuracy'] = recent_scores['accuracy'].apply(lambda x: float(x.replace("%", "")) / 100)
            X = recent_scores[['score', 'accuracy', 'total_questions']]
            y = df_historical_quiz['rank']  # Assume historical rank data is available in the dataset

            # Machine Learning Model: Linear Regression for Rank Prediction
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            print(f"Mean Absolute Error in NEET Rank Prediction: {mae}")
            print(f"Predicted NEET Rank: {y_pred[0]}")

# Run the analysis
analyze_performance(df_current_quiz, df_current_metadata, df_historical_quiz)
