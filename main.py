import streamlit as st
import torch
import re
from pymystem3 import Mystem
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast
from nltk import word_tokenize
from nltk import download as nltk_load

@torch.no_grad()
def predict(text):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(predicted, dim=1).numpy()
    return predicted

is_punkt_load = False

def punkt_load():
    global is_punkt_load

    if is_punkt_load:
        return

    nltk_load('punkt')

    is_punkt_load = True

def text_prepocessing(text):

    text = text.lower()

    text = re.sub(r'[^\u0400-\u04FF]', ' ', text)
    punkt_load()
    text_tokens = word_tokenize(text)

    text_tokens = [stem.lemmatize(token)[0] for token in text_tokens]
    text = ' '.join(text_tokens)

    return text
    

def do_analysis(text):

    if text == '' or text == None:
        return "Ждём ввод 🕒"

    result = predict(text_prepocessing(text))
    
    if result == 1:
        result = "Положительно 🙂"
    if result == 0:
        result = "Нейтрально 😐"
    if result == 2:
        result = "Отрицательно ☹️"

    return result


stem = Mystem()

tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment-rurewiews')
model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment-rurewiews', return_dict=True)

st.title("Анализ тональности отзывов о товарах")

st.write("""
        Введите отзыв в поле ниже и увидьте его тональность.
         
        Программа работает для отзывов, написанных на русском языке.
        """)

text = st.text_area(label="Я поле ниже", placeholder="Отзыв сюда")

st.write(f"##### Результат: {do_analysis(text)}")
