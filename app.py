import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re

st.set_page_config(page_title='Trust Guard AI', layout='wide')

if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

def get_text_column(df):
    """Find text column - tries common names with case-insensitive matching"""
    text_cols = ['review', 'text', 'review_text', 'comment', 'feedback', 'description', 'body', 'message']
    cols_lower = {col.lower(): col for col in df.columns}
    
    # First try exact matches (case-insensitive)
    for text_col in text_cols:
        if text_col in cols_lower:
            return cols_lower[text_col]
    
    # Then try partial matches for 'review' or 'text' in column names
    for col in df.columns:
        col_lower = col.lower()
        if 'review' in col_lower and df[col].dtype == 'object':
            return col
    
    # Try 'text' in column names
    for col in df.columns:
        col_lower = col.lower()
        if 'text' in col_lower and df[col].dtype == 'object':
            return col
    
    # If no common name found, return first text column
    for col in df.columns:
        if df[col].dtype == 'object' and col.lower() not in ['rating', 'rate', 'score', 'name', 'user']:
            return col
    
    raise ValueError(f'No text column found. Columns: {list(df.columns)}')

def get_rating_column(df):
    """Find rating column - tries common names with case-insensitive matching"""
    rating_cols = ['rating', 'rate', 'score', 'stars', 'overall']
    cols_lower = {col.lower(): col for col in df.columns}
    
    # First try exact matches (case-insensitive)
    for rating_col in rating_cols:
        if rating_col in cols_lower:
            return cols_lower[rating_col]
    
    # Then try partial matches
    for col in df.columns:
        col_lower = col.lower()
        if 'rating' in col_lower and df[col].dtype in ['int64', 'float64']:
            return col
    
    for col in df.columns:
        col_lower = col.lower()
        if 'overall' in col_lower and df[col].dtype in ['int64', 'float64']:
            return col
    
    # If no common name found, return first numeric column
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            return col
    
    raise ValueError(f'No rating column found. Columns: {list(df.columns)}')

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentiment_label(polarity):
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def detect_fake_review(cleaned_text, polarity, subjectivity, rating, sentiment):
    fake_score = 0
    text_words = len(cleaned_text.split())
    if text_words < 10:
        fake_score += 0.2
    if polarity == 0:
        fake_score += 0.15
    if subjectivity > 0.9:
        fake_score += 0.2
    if (rating >= 4 and sentiment == 'Negative') or (rating <= 2 and sentiment == 'Positive'):
        fake_score += 0.25
    return min(fake_score, 1.0)

def calculate_trust_score(fake_score, sentiment, is_fake):
    trust_score = 100
    trust_score -= (fake_score * 40)
    if sentiment == 'Negative':
        trust_score -= 20
    elif sentiment == 'Neutral':
        trust_score -= 10
    if not is_fake:
        trust_score += 15
    return max(0, min(trust_score, 100))

def analyze_reviews(df, text_col, rating_col):
    result_df = df.copy()
    result_df['cleaned_text'] = result_df[text_col].apply(preprocess_text)
    result_df['polarity'] = result_df[text_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    result_df['subjectivity'] = result_df[text_col].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    result_df['sentiment'] = result_df['polarity'].apply(get_sentiment_label)
    
    result_df['fake_score'] = result_df.apply(
        lambda row: detect_fake_review(row['cleaned_text'], row['polarity'], row['subjectivity'], row[rating_col], row['sentiment']),
        axis=1
    )
    result_df['is_fake'] = result_df['fake_score'] >= 0.5
    result_df['trust_score'] = result_df.apply(
        lambda row: calculate_trust_score(row['fake_score'], row['sentiment'], row['is_fake']),
        axis=1
    )
    return result_df

page = st.sidebar.radio('Navigation', ['Home', 'Upload & Analyze', 'Dashboard', 'Detailed Analysis', 'About'])

if page == 'Home':
    st.title('Trust Guard AI')
    st.subheader('Customer Review Analysis System')
    st.write('Welcome to Trust Guard AI, an intelligent system for analyzing and validating customer reviews.')
    st.info('Features: Sentiment Analysis, Fake Review Detection, Trust Scoring')

elif page == 'Upload & Analyze':
    st.title('Upload & Analyze Reviews')
    uploaded_file = st.file_uploader('Upload CSV file', type='csv')
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f'CSV Columns found: {list(df.columns)}')
            text_col = get_text_column(df)
            rating_col = get_rating_column(df)
            st.write(f'Using text column: {text_col}, rating column: {rating_col}')
            st.write(f'Loaded {len(df)} reviews')
            df = analyze_reviews(df, text_col, rating_col)
            st.session_state.uploaded_data = df
            st.session_state.text_col = text_col
            st.session_state.rating_col = rating_col
            st.success('Analysis complete!')
            st.dataframe(df[[text_col, rating_col, 'sentiment', 'trust_score']])
        except Exception as e:
            st.error(f'Error: {str(e)}')

elif page == 'Dashboard':
    st.title('Dashboard')
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        rating_col = st.session_state.get('rating_col', 'rating')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Total Reviews', len(df))
        with col2:
            st.metric('Avg Trust Score', f"{df['trust_score'].mean():.2f}")
        with col3:
            st.metric('Fake Reviews', df['is_fake'].sum())
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            df['sentiment'].value_counts().plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Sentiment Distribution')
            axes[0, 1].hist(df['trust_score'], bins=10)
            axes[0, 1].set_title('Trust Score Distribution')
            axes[1, 0].scatter(df[rating_col], df['trust_score'])
            axes[1, 0].set_title(f'{rating_col} vs Trust Score')
            axes[1, 1].hist(df['polarity'], bins=20)
            axes[1, 1].set_title('Polarity Distribution')
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f'Visualization error: {str(e)}')
    else:
        st.warning('Please upload data first!')

elif page == 'Detailed Analysis':
    st.title('Detailed Review Analysis')
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        st.dataframe(df, height=400)
    else:
        st.warning('Please upload data first!')

elif page == 'About':
    st.title('About Trust Guard AI')
    st.write('AI system for analyzing customer reviews using Natural Language Processing')
    st.write('Built for Saveetha AI Institution')
