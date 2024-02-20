import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

# Download stopwords bahasa Indonesia jika belum diunduh
nltk.download('punkt')
nltk.download('stopwords')

# Fungsi pra-pemrosesan teks
def preprocess_text(text):
    # tambahkan pra-pemrosesan teks Anda di sini
    return text

# Fungsi untuk melakukan tokenisasi teks
def tokenize_text(text):
    tokens = word_tokenize(text)  # Melakukan tokenisasi menggunakan NLTK
    return tokens

# Fungsi untuk menghapus stopwords dari teks
def remove_stopwords(text):
    stop_words_indonesian = set(stopwords.words('indonesian'))
    tokens = text.split()  # Memecah teks menjadi token
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words_indonesian]  # Menghapus stopwords
    return ' '.join(filtered_tokens)  # Menggabungkan token menjadi teks kembali

# Fungsi untuk melakukan stemming
def stem_text(text):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split()])

# Memuat model
clf = joblib.load('svm.joblib')
# Memuat objek vectorizer
vectorizer = joblib.load('vectorizer.joblib')
# Memuat objek TfidfTransformer
tfidf_transformer = joblib.load('tfidf_transformer.joblib')

# Tampilan aplikasi Streamlit
st.title("Aplikasi Analisis Sentimen")

# Masukkan teks dari pengguna
input_text = st.text_area("Masukkan teks:")

if st.button("Analisis Sentimen"):
    # Pra-pemrosesan teks
    processed_text = preprocess_text(input_text)
    # Tokenisasi teks
    tokenized_text = tokenize_text(processed_text)
    # Hapus stopwords
    text_without_stopwords = remove_stopwords(processed_text)
    # Lakukan stemming
    stemmed_text = stem_text(text_without_stopwords)
    
    # Transformasi teks menjadi vektor menggunakan vectorizer yang telah dimuat
    text_vectorized = vectorizer.transform([stemmed_text])
    # Transformasi vektor menggunakan TfidfTransformer yang telah dimuat
    text_tfidf = tfidf_transformer.transform(text_vectorized)
    
    # Mengonversi data teks yang telah diproses menjadi array 2D
    text_tfidf_array = text_tfidf.toarray()
    
    # Lakukan prediksi sentimen
    prediction = clf.predict(text_tfidf_array)
    
    # Tampilkan hasil prediksi
    if prediction == -1:
        st.write("Sentimen: Negatif")
    elif prediction == 0:
        st.write("Sentimen: Netral")
    else:
        st.write("Sentimen: Positif")
