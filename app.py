import streamlit as st
import pandas as pd
from textblob import TextBlob
import re

st.set_page_config(page_title='Trust Guard AI', layout='wide')

if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
    st.session_state.text_col = None
    st.session_state.rating_col = None

def get_text_column(df):
    """Find text column - tries common names with case-insensitive matching"""
    text_cols = ['review', 'text', 'review_text', 'comment', 'feedback', 'description', 'body', 'message']
    cols_lower = {col.lower(): col for col in df.columns}
    
    for text_col in text_cols:
        if text_col in cols_lower:
            return cols_lower[text_col]
    
    for col in df.columns:
        col_lower = col.lower()
        if 'review' in col_lower and df[col].dtype == 'object':
            return col
    
    for col in df.columns:
        col_lower = col.lower()
        if 'text' in col_lower and df[col].dtype == 'object':
            return col
    
    for col in df.columns:
        if df[col].dtype == 'object' and col.lower() not in ['rating', 'rate', 'score', 'name', 'user']:
            return col
    
    raise ValueError(f'No text column found. Columns: {list(df.columns)}')

def get_rating_column(df):
    """Find rating column - tries common names with case-insensitive matching"""
    rating_cols = ['rating', 'rate', 'score', 'stars', 'overall']
    cols_lower = {col.lower(): col for col in df.columns}
    
    for rating_col in rating_cols:
        if rating_col in cols_lower:
            return cols_lower[rating_col]
    
    for col in df.columns:
        col_lower = col.lower()
        if 'rating' in col_lower and df[col].dtype in ['int64', 'float64']:
            return col
    
    for col in df.columns:
        col_lower = col.lower()
        if 'overall' in col_lower and df[col].dtype in ['int64', 'float64']:
            return col
    
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
        text_col = st.session_state.text_col or 'text'
        rating_col = st.session_state.rating_col or 'rating'
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Total Reviews', len(df))
        with col2:
            st.metric('Avg Trust Score', f"{df['trust_score'].mean():.2f}")
        with col3:
            st.metric('Fake Reviews', df['is_fake'].sum())
        
        # Sentiment Distribution
        st.subheader('Sentiment Distribution')
        sentiment_counts = df['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
        
        # Trust Score Distribution  
        st.subheader('Trust Score Distribution')
        trust_bins = pd.cut(df['trust_score'], bins=5).value_counts().sort_index()
        st.bar_chart(trust_bins)
        
        # Polarity Distribution
        st.subheader('Polarity Distribution')
        polarity_bins = pd.cut(df['polarity'], bins=5).value_counts().sort_index()
        st.bar_chart(polarity_bins)
        
        # Fake Review Distribution
        st.subheader('Fake Review Count')
        fake_counts = df['is_fake'].value_counts()
        fake_data = pd.DataFrame({'Category': ['Authentic', 'Suspicious'], 'Count': [fake_counts[False] if False in fake_counts.index else 0, fake_counts[True] if True in fake_counts.index else 0]})
        st.bar_chart(fake_data.set_index('Category'))
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
