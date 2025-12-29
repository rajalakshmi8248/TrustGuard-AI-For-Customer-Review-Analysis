# Trust Guard AI - Customer Review Analysis System

## Overview
Trust Guard AI is an intelligent AI-driven system for analyzing and validating online customer reviews using Natural Language Processing (NLP) and Machine Learning techniques.

The system performs comprehensive analysis of product/service reviews to detect sentiment, identify potentially fake reviews, and calculate a trust score for each review.

## Features

- **Sentiment Analysis**: Uses TextBlob to analyze polarity and subjectivity of reviews
- **Fake Review Detection**: Multi-feature algorithm that identifies suspicious reviews based on:
  - Text length and word count
  - All-caps spam indicators
  - Extreme subjectivity scores
  - Rating-sentiment misalignment
  - Suspicious patterns
- **Trust Score Calculation**: Generates a 0-100 trust score based on multiple factors
- **Visual Analytics**: Interactive dashboards with charts and statistics
- **Easy Deployment**: Streamlit-based web application

## Technology Stack

- **Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **NLP**: TextBlob, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Python**: 3.8+

## Installation

### Local Setup

```bash
# Clone the repository
git clone https://github.com/rajalakshmi8248/TrustGuard-AI-For-Customer-Review-Analysis.git
cd TrustGuard-AI-For-Customer-Review-Analysis

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The application will open at `http://localhost:8501`

### Cloud Deployment (Streamlit Cloud - Recommended)

1. Create a GitHub account and fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository
6. Set main file as `app.py`
7. Click "Deploy"

## Usage

1. **Home Page**: Introduction and feature overview
2. **Upload & Analyze**: Upload CSV file with columns `review_text` and `rating`
3. **Dashboard**: View statistics and visualizations
4. **Detailed Analysis**: Filter and examine individual reviews
5. **About**: Project information

## CSV Format

Upload a CSV file with the following structure:

```
review_text,rating
"Great product, very satisfied!",5
"Poor quality, waste of money",1
```

## Output Metrics

- **Sentiment**: Positive, Negative, or Neutral
- **Polarity Score**: -1.0 to 1.0 (negative to positive)
- **Subjectivity Score**: 0.0 to 1.0 (objective to subjective)
- **Fake Score**: 0.0 to 1.0 (likelihood of being fake)
- **Trust Score**: 0-100 (overall trustworthiness)

## Project Structure

```
trust-guard-ai/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Author

Developed by: Saveetha AI Student
Institution: Saveetha AI

## License

This project is licensed under the GPL-3.0 License - see the LICENSE.txt file for details.

## Acknowledgments

- TextBlob for NLP capabilities
- Streamlit for web framework
- Scikit-learn for machine learning
