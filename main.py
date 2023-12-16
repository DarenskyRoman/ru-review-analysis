import streamlit as st
import re
from transformers import pipeline
from pymystem3 import Mystem
from razdel import tokenize
#from torch import no_grad
#from torch import argmax
#from torch.nn.functional import softmax
#from transformers import AutoModelForSequenceClassification
#from transformers import BertTokenizerFast



#@no_grad()
#def predict(text):
#    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
#    outputs = model(**inputs)
#    predicted = softmax(outputs.logits, dim=1)
#   predicted = argmax(predicted, dim=1).numpy()
#    return predicted

def predict(text):
    return pipe(text)

def text_prepocessing(text):

    text = text.lower()

    text = re.sub(r'[^\u0400-\u04FF]', ' ', text)

    text_tokens = list(tokenize(text))
    text_tokens = [_.text for _ in text_tokens]
    
    text_tokens = [word for word in text_tokens if len(word) > 1]
    text_tokens = [stem.lemmatize(token)[0] for token in text_tokens]
    text = ' '.join(text_tokens)

    return text
    
def remove_chars_from_text(text, chars):
    return "".join([ch for ch in text if ch not in chars])

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

#tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment-rurewiews')
#model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment-rurewiews', return_dict=True)

pipe = pipeline("text-classification", model="blanchefort/rubert-base-cased-sentiment-rurewiews")



st.title("Анализ тональности отзывов о товарах")

st.write("""
        Введите отзыв в поле ниже и увидьте его тональность.
         
        Программа работает для отзывов, написанных на русском языке.
        """)

text = st.text_area(label="Я поле ниже", placeholder="Отзыв сюда")

st.write(f"##### Результат: {do_analysis(text)}")
