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
def ngramming(df_positive, df_neutral, df_negative):
    df_positive = " ".join(df_positive)
    df_neutral = " ".join(df_neutral)
    df_negative = " ".join(df_negative)

    df_positive = text_prepocessing(df_positive, do_lemm=False)
    df_neutral = text_prepocessing(df_neutral, do_lemm=False)
    df_negative = text_prepocessing(df_negative, do_lemm=False)

    st.write("Позитивные 🙂 n-граммы:\n")
    ngramm_extraction(df_positive)
    st.write("Нейтральные 😐 n-граммы:\n")
    ngramm_extraction(df_neutral)
    st.write("Негативные ☹️ n-граммы:\n")
    ngramm_extraction(df_negative)

@st.cache_data
def ngramm_extraction(text):
    one = list(ngrams(text.split(), 1))
    two = list(ngrams(text.split(), 2))
    three = list(ngrams(text.split(), 3))

    fdist_one = FreqDist(one).most_common(5)
    fdist_two = FreqDist(two).most_common(5)
    fdist_three = FreqDist(three).most_common(5)

    fdist_one = ''.join(f" {' '.join(item[0])} - {item[1]} | " for item in fdist_one)
    fdist_two = ''.join(f" {' '.join(item[0])} - {item[1]} | " for item in fdist_two)
    fdist_three = ''.join(f" {' '.join(item[0])} - {item[1]} | " for item in fdist_three)

    st.write("Униграммы:")
    st.write(fdist_one)

    st.write("Биграммы:")
    st.write(fdist_two)

    st.write("Триграммы:")
    st.write(fdist_three)

@st.cache_data
def file_analisys(file, reviews_number, column_name, shuffle, do_ngramms):
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
            ngramming(df_positive, df_neutral, df_negative)

        return df.to_csv()

    except:
        st.warning("Не можем прочитать файл ☹️. Всё ли вы ввели верно?")
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
        а также (по желанию) набор часто встречающихся 1, 2 и 3-грамм.
        """)

file = st.file_uploader(label="Я загружаю файлы", type="csv")

st.write("""
        ###### Перед анализом загруженных выражений необходимо кое-что определить ниже:
        """)

reviews_number = st.number_input(label="Количество выражений для анализа", min_value=1, max_value=1000, value="min", step=1)
column_name = st.text_input(label="Название столбца с выражениями", value="gugu gaga")
shuffle = st.checkbox(label="Перемешиваем? P.S. Полезно, если выражения в датасете сгруппированны по тональности (идут подряд)")
do_ngramms = st.checkbox(label="Выводим n-граммы?")

if check_prep() == 0:
    st.button(label="Анализировать", disabled=True)
else:
    bupton = st.button(label="Анализировать", disabled=False)
    if bupton:
        file_result = file_analisys(file, reviews_number, column_name, shuffle, do_ngramms)
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
