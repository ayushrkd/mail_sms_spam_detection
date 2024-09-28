from copyreg import pickle


import streamlit as st
import string
# nltk.download('stopwords')
import sklearn
from nltk.corpus import  stopwords
from nltk.parse.corenlp import transform
import dill as pickle
# stopwords.words("english")
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


tfidf = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

def text_convert1(text):
    text = text.lower()  # convertint into lower case
    text = text.split()  # tokenizer word split -->list

    y = []
    for i in text:
        if i.isalnum():  # removing special char--> @ & %
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if (i not in stopwords.words("english")) and (i not in string.punctuation):
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


st.title("Mail spam detector system")

input_text=st.text_area(label="type your message")

if st.button(label="predict"):
    # transform text
    transformed_text = text_convert1(input_text)
    # 2. vectorize
    vector_input=tfidf.transform([transformed_text])
    # 3. model
    result = model.predict(vector_input)[0]

    if result ==0:
        st.header('Not Spam')
    else:
        st.header('Spam')


        # example
        # congratulations you won a 1000 call on this number to get your prize
        # spam