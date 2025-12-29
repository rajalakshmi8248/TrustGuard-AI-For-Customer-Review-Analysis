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

def analyze_reviews(df):
    result_df = df.copy()
    result_df['cleaned_text'] = result_df['review_text'].apply(preprocess_text)
    result_df['polarity'] = result_df['review_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    result_df['subjectivity'] = result_df['review_text'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    result_df['sentiment'] = result_df['polarity'].apply(get_sentiment_label)
    
    result_df['fake_score'] = result_df.apply(
        lambda row: detect_fake_review(row['cleaned_text'], row['polarity'], row['subjectivity'], row['rating'], row['sentiment']),
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
            st.write(f'Loaded {len(df)} reviews')
            df = analyze_reviews(df)
            st.session_state.uploaded_data = df
            st.success('Analysis complete!')
            st.dataframe(df[['review_text', 'rating', 'sentiment', 'trust_score']])
        except Exception as e:
            st.error(f'Error: {str(e)}')

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
        
        try:
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
