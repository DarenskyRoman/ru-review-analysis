import streamlit as st
import re
import pandas as pd
from nltk.probability import FreqDist
from nltk.util import ngrams
from pymystem3 import Mystem
from razdel import tokenize
from transformers import pipeline


@st.cache_data
def predict(text):
    mapping = {"neutral": 0, "positive": 1, "negative": 2}
    result = mapping[model(text)[0]["label"]]
    return result

@st.cache_data
def text_prepocessing(text, do_lemm=False):
    text = text.lower()
    text = re.sub(r'[^\u0400-\u04FF]', ' ', text)
    text = text.replace('ё', 'е')

    text_tokens = list(tokenize(text))
    text_tokens = [_.text for _ in text_tokens]
    
    text_tokens = [word for word in text_tokens if len(word) > 1]
    if do_lemm:
        text_tokens = [stem.lemmatize(token)[0] for token in text_tokens]
    text = ' '.join(text_tokens)

    return text


@st.cache_data
def do_analysis(text):
    if text == '' or text == None:
        return "Ждём ввод 🕒"

    result = predict(text_prepocessing(text))
    
    if result == 1:
        result = "Положительно 🙂"
    if result == 0:
        result = "Нейтрально 😐"
    if result == 2:
        result = "Негативненько ☹️"

    return result


@st.cache_data
def ngramming(df_positive, df_neutral, df_negative, ngramm_power, ngramm_count):
    df_positive = " ".join(df_positive)
    df_neutral = " ".join(df_neutral)
    df_negative = " ".join(df_negative)

    df_positive = text_prepocessing(df_positive, do_lemm=True)
    df_neutral = text_prepocessing(df_neutral, do_lemm=True)
    df_negative = text_prepocessing(df_negative, do_lemm=True)
    
    st.write(f"Позитивные 🙂 {ngramm_power}-граммы:\n")
    ngramm_extraction(df_positive, ngramm_power, ngramm_count)
    st.write(f"Нейтральные 😐 {ngramm_power}-граммы:\n")
    ngramm_extraction(df_neutral, ngramm_power, ngramm_count)
    st.write(f"Негативные ☹️ {ngramm_power}-граммы:\n")
    ngramm_extraction(df_negative, ngramm_power, ngramm_count)

@st.cache_data
def ngramm_extraction(text, ngramm_power, ngramm_count):
    ng = list(ngrams(text.split(), ngramm_power))

    fdist_ng = FreqDist(ng).most_common(ngramm_count)
    fdist_ng = ''.join(f" {' '.join(item[0])} - {item[1]} | " for item in fdist_ng)
    st.write(fdist_ng)

@st.cache_data
def file_analisys(file, reviews_number, column_name, shuffle, do_ngramms=False, ngramm_power=None, ngramm_count=None):
    try:
        if shuffle:
            df = pd.read_csv(file, usecols=[column_name]).sample(n=reviews_number)
        else:
            df = pd.read_csv(file, usecols=[column_name], nrows=reviews_number)

        df["predict"] = df["review"].apply(predict)

        df_positive = df["review"][df["predict"] == 1]
        df_neutral = df["review"][df["predict"] == 0]
        df_negative = df["review"][df["predict"] == 2]

        st.write(df.head(5))

        st.write(f"Количество позитивных 🙂 выражений: {df_positive.count()}")
        st.write(f"Количество нейтральных 😐 выражений: {df_neutral.count()}")
        st.write(f"Количество негативных ☹️ выражений: {df_negative.count()}")

        if do_ngramms:
            ngramming(df_positive, df_neutral, df_negative, ngramm_power, ngramm_count)

        return df.to_csv()

    except:
        st.warning("Что-то не получается ☹️. Всё ли вы ввели верно?")
        return "Онет"

@st.cache_resource
def load_model():
    return pipeline(model="seara/rubert-tiny2-russian-sentiment")

@st.cache_resource
def load_lemmatizer():
    return Mystem()

def check_prep():
    if file == None or (column_name == '' or column_name == None):
        return 0
    else:
        return 1


stem = load_lemmatizer()
model = load_model()

st.title("Анализ тональности выражений")

st.write("\n\n")
st.write("\n\n ##### Одно выражение")

st.write("""
        Введите выражение в поле ниже и увидьте его тональность.
         
        Программа работает для выражений, написанных на русском языке.
        """)

text = st.text_area(label="Я поле ниже", placeholder="Отзыв сюда")

st.write(f"##### Результат: {do_analysis(text)}")

st.write("\n\n")
st.write("\n\n ##### Много выражений")

st.write("""
        Ещё вы можете загрузить датасет выражений в формате csv.
         
        Программа выведет количество положительных, отрицательных и нейтральных выражений,
        а также (по желанию) набор часто встречающихся n-грамм.
        """)

file = st.file_uploader(label="Я загружаю файлы", type="csv")

st.write("""
        ###### Перед анализом загруженных выражений необходимо кое-что определить ниже:
        """)

reviews_number = st.number_input(label="Количество выражений для анализа", min_value=1, max_value=1000, value="min", step=1)
column_name = st.text_input(label="Название столбца с выражениями", value="gugu gaga")
shuffle = st.checkbox(label="Перемешиваем? P.S. Полезно, если выражения в датасете сгруппированны по тональности (идут подряд)")
do_ngramms = st.checkbox(label="Выводим популярные n-граммы?")
if do_ngramms:
    ngramm_power = st.number_input(label="Степень n-грамм", min_value=1, max_value=5, value="min", step=1)
    ngramm_count = st.number_input(label="Сколько выводим?", min_value=1, max_value=10, value="min", step=1)

if check_prep() == 0:
    st.button(label="Анализировать", disabled=True)
else:
    bupton = st.button(label="Анализировать", disabled=False)
    if bupton:
        if 'ngramm_power' in locals():
            file_result = file_analisys(file, reviews_number, column_name, shuffle, do_ngramms, ngramm_power, ngramm_count)
        else:
            file_result = file_analisys(file, reviews_number, column_name, shuffle)

        if file_result != "Онет":
            st.write("""
                ###### Хотите скачать получившийся файл?
                """)
            st.download_button(
                label="Да, хочу",
                data=file_result,
                file_name='result.csv',
                mime='text/csv',
            )
