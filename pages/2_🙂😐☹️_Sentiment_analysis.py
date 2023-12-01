import streamlit as st
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel


st.set_page_config(
    page_title="Sentiment analysis"
)

def do_analysis(text):

    if text == '' or text == None:
        return "we are waiting for input 🕒"

    messages = [text]

    result = model.predict(messages, k=2)
    result = list(result[0].keys())[0]
    if result == "positive":
        result = "emotional tone - positive 😐"
    if result == "neutral":
        result = "emotional tone - neutral 😐"
    if result == "negative":
        result = "emotional tone - negative 😐"
    if result == "speech":
        result = "we didn't find any emotions in this text 🗿"
    if result == "skip":
        result = "we can't determine the emotional tone of this text, sorry 🐹"

    return result

tokenizer = RegexTokenizer()
model = FastTextSocialNetworkModel(tokenizer=tokenizer)

st.title("Sentiment analysis")

st.write("""
        Input your text in a cell bellow and see evaluation of the emotional tone of your text.
        """)

text = st.text_input(label="I'm a text cell", placeholder="Write your text here")

st.write(f"##### Result: {do_analysis(text)}")
