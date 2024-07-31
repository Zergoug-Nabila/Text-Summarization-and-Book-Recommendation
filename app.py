import streamlit as st
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Define model paths
models = {
    'English': {
        'model_path': 'saved_model_en',
        'tokenizer_path': 'saved_tokenizer_en'
    },
    'French': {
        'model_path': 'saved_model_fr',
        'tokenizer_path': 'saved_tokenizer_fr'
    }
}

# Load model and tokenizer
def load_model_and_tokenizer(lang):
    try:
        model_path = models[lang]['model_path']
        tokenizer_path = models[lang]['tokenizer_path']
        
        model = BartForConditionalGeneration.from_pretrained(model_path)
        tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer for {lang}: {e}")
        return None, None

def summarize_text(text, model, tokenizer):
    try:
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return ""

# Book recommendation function
def recommend_books(df, summary, top_n=2):
    try:
        # Extract keywords from summary
        vectorizer = TfidfVectorizer(stop_words='english')
        summary_vec = vectorizer.fit_transform([summary])

        # Calculate similarity between summary and book titles
        df['similarity'] = df['title'].apply(lambda x: cosine_similarity(vectorizer.transform([x]), summary_vec)[0][0])

        # Select top N books
        recommended_books = df.nlargest(top_n, 'similarity')
        return recommended_books[['title', 'authors', 'average_rating']]
    except Exception as e:
        st.error(f"Error recommending books: {e}")
        return pd.DataFrame(columns=['title', 'authors', 'average_rating'])

# App title
st.markdown("<h1 style='font-size: 2em;'>Book Recommendation System</h1>", unsafe_allow_html=True)

# Load book data
try:
    df = pd.read_csv("data_preprocessed.csv")
except Exception as e:
    st.error(f"Error loading book data: {e}")
    df = pd.DataFrame(columns=['title', 'authors', 'average_rating'])

# Choose language
st.markdown("<h2 style='font-size: 1.5em; font-weight: bold;'>Choose language</h2>", unsafe_allow_html=True)
lang = st.selectbox("Language", ['English', 'French'])

# Load model and tokenizer for the chosen language
model, tokenizer = load_model_and_tokenizer(lang)

if model and tokenizer:
    # Input text
    st.markdown("<h2 style='font-size: 1.5em; font-weight: bold;'>Enter text to summarize</h2>", unsafe_allow_html=True)
    text = st.text_area("Input Text", placeholder="Type your text here...", height=200) 

    if st.button("Summarize and Recommend"):
        if text:
            # Generate summary
            summary = summarize_text(text, model, tokenizer)
            st.markdown("<h2 style='font-size: 1.5em; font-weight: bold;'>Summary:</h2>", unsafe_allow_html=True)
            st.write(summary)
            
            # Recommend books
            recommended_books = recommend_books(df, summary)
            st.markdown("<h2 style='font-size: 1.5em; font-weight: bold;'>Recommended Books:</h2>", unsafe_allow_html=True)
            st.dataframe(recommended_books)
        else:
            st.warning("Please enter some text.")
else:
    st.warning("Models could not be loaded.")

# Footer
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        box-shadow: 0 -1px 5px rgba(0,0,0,0.1);
        cursor: pointer;
        background-color: #3f4059;
    }
    </style>
    <div class="footer" onclick="this.style.display='none'">
        <p>@ZERGOUG Nabila</p>
    </div>
""", unsafe_allow_html=True)
