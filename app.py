
import streamlit as st
import pickle


st.set_page_config(page_title=" Fake News Detector", page_icon="Fake News Detector", layout="centered")


model = pickle.load(open(r"C:\Users\ASUS\OneDrive\Desktop\SATISH PROJECTS\FAKE NEWS\model.pkl", "rb"))
vectorizer = pickle.load(open(r"C:\Users\ASUS\OneDrive\Desktop\SATISH PROJECTS\FAKE NEWS\vectorizer.pkl", "rb"))

st.markdown("""
    <style>
        body {
            background-color: #111111;
            color: white;
        }
        h1 {
            font-size: 50px !important;
            font-weight: 900 !important;
            color: white !important;
        }
        p {
            font-size: 20px !important;
            color: #cccccc !important;
            font-weight: 500 !important;
        }
    </style>
""", unsafe_allow_html=True)




st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='font-size: 48px; color: #e63946;'>Fake News Detection System</h1>
        <p style='font-size: 18px;'>Detect whether a news article is <strong>Real</strong> or <strong>Fake</strong> using Machine Learning</p>
    </div>
""", unsafe_allow_html=True)






news_text = st.text_area(" Enter the News Text Below", height=200, placeholder="Type or paste the news article here...")


if st.button("Predict 🔎"):
    if news_text.strip() == "":
        st.warning("Please enter some news text to check.")
    else:
        input_data = vectorizer.transform([news_text])
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("✅ This news is REAL.")
            st.markdown("🟢 **Our ML model believes this is a trustworthy news article.**")
        else:
            st.error("🚨 This news is FAKE.")
            st.markdown("🔴 **Our ML model suspects this news might be misleading. Be cautious!**")


st.markdown("---")
st.markdown("<center>Made by Satish Dwivedi | Powered by Machine Learning</center>", unsafe_allow_html=True)
