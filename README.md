# Project-quiz

NEET Testline - Personalized Student Recommendations
Project Overview
NEET Testline is a Python-based solution that analyzes students' quiz performance and provides personalized recommendations to improve their preparation. The system processes both current and historical quiz data to generate insights, highlight weak areas, and suggest actionable study plans. Additionally, a machine learning model predicts the student's estimated NEET rank based on quiz accuracy trends.
Features
Performance Analysis: Evaluate quiz accuracy, score progression, and topic-wise strengths/weaknesses.
Personalized Study Recommendations: Suggest topics and question types based on past performance.
NEET Rank Prediction: Estimate rank using a linear regression model based on historical quiz data.
Interactive Dashboard: A front-end interface with data visualization and study insights.

Setup Instructions
Prerequisites
Ensure you have the following installed:
Python 3.8+
pip
Node.js & npm (for frontend modifications)
Backend Setup
Clone the Repository
 git clone https://github.com/your-repo-url.git
cd NEET-Testline
Create and Activate Virtual Environment
 python -m venv venv source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies
 pip install -r requirements.txt
Ensure Data Files Exist
Place current_quiz.json and historical_quiz.json inside the data/ directory.

Run the Analysis
 python backend.py

Frontend Setup
Open the index.html file in a browser.
Ensure an active connection to the backend if using dynamic data loading.

Approach & Methodology
1. Data Collection
Fetch current quiz submission data from current_quiz.json.
Retrieve last 5 quizzes from historical_quiz.json.
2. Data Processing & Analysis
Convert JSON data into Pandas DataFrames.
Extract quiz metadata: accuracy, scores, correct/incorrect responses.
Identify weak topics based on incorrect answers.
Analyze historical accuracy and progression trends.
3. Machine Learning Model
Use Linear Regression to predict NEET rank.
Train model using features: score, accuracy, total questions.
Evaluate performance with Mean Absolute Error (MAE).
4. Personalized Recommendations
Generate topic-wise study suggestions based on performance gaps.
Highlight specific weak subjects with actionable advice.
Provide study plan recommendations tailored to the student.

Future Enhancements
Implement Adaptive Testing for real-time quiz difficulty adjustment.
Use Deep Learning Models for better rank prediction accuracy.
Develop a Mobile App Integration for seamless access to recommendations.

Video explanation
https://drive.google.com/file/d/1M-GZ_IvpqHMRiCzO7bopsART56TR7oFx/view?usp=sharing
