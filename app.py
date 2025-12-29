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

def preprocess_text(text):
    text = text.lower()
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

def detect_fake_review(row):
    fake_score = 0
    text_words = len(row['cleaned_text'].split())
    if text_words < 10:
        fake_score += 0.2
    if row['polarity'] == 0:
        fake_score += 0.15
    if row['subjectivity'] > 0.9:
        fake_score += 0.2
    if (row['rating'] >= 4 and row['sentiment'] == 'Negative') or (row['rating'] <= 2 and row['sentiment'] == 'Positive'):
        fake_score += 0.25
    return min(fake_score, 1.0)

def calculate_trust_score(row):
    trust_score = 100
    trust_score -= (row['fake_score'] * 40)
    if row['sentiment'] == 'Negative':
        trust_score -= 20
    elif row['sentiment'] == 'Neutral':
        trust_score -= 10
    if not row['is_fake']:
        trust_score += 15
    return max(0, min(trust_score, 100))

def analyze_reviews(df):
    df['cleaned_text'] = df['review_text'].apply(preprocess_text)
    df['polarity'] = df['review_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['review_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df['sentiment'] = df['polarity'].apply(get_sentiment_label)
    df['fake_score'] = df.apply(detect_fake_review, axis=1)
    df['is_fake'] = df['fake_score'] >= 0.5
    df['trust_score'] = df.apply(calculate_trust_score, axis=1)
    return df

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
        df = pd.read_csv(uploaded_file)
        st.write(f'Loaded {len(df)} reviews')
        df = analyze_reviews(df)
        st.session_state.uploaded_data = df
        st.success('Analysis complete!')
        st.dataframe(df[['review_text', 'rating', 'sentiment', 'trust_score']])

elif page == 'Dashboard':
    st.title('Dashboard')
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Total Reviews', len(df))
        with col2:
            st.metric('Avg Trust Score', f"{df['trust_score'].mean():.2f}")
        with col3:
            st.metric('Fake Reviews', df['is_fake'].sum())
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        df['sentiment'].value_counts().plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Sentiment Distribution')
        axes[0, 1].hist(df['trust_score'], bins=10)
        axes[0, 1].set_title('Trust Score Distribution')
        axes[1, 0].scatter(df['rating'], df['trust_score'])
        axes[1, 0].set_title('Rating vs Trust Score')
        axes[1, 1].hist(df['polarity'], bins=20)
        axes[1, 1].set_title('Polarity Distribution')
        plt.tight_layout()
        st.pyplot(fig)
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
