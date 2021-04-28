import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fbprophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from lppls import lppls, data_loader


st.title('BITCOIN PRICE ANALYSIS')
st.header("Disclaimer!")
st.markdown("""This is not Financial Advice and should not be used as the only reference
to making your final decision. The material is meant for Educational Purposes. Furthermore, this data
is as of April 2021.""")
st.header("Is It A Bubble Or Not?")
st.markdown("""There are 2 models to help determine this, FaceBook Prophet and Log Periodic
Power Law Singularity (LPPLS) model. We can view either of the 2 by selecting from the sidebar on
the left""")

st.sidebar.header('Select Model')
model_select = st.sidebar.selectbox("Model", options=["LPPLS","fb-Prophet"])


bitcoin = pd.read_csv("numeric_clean_bitcoin_data.csv")
bitcoin = bitcoin.drop(columns=["Unnamed: 0"])
bitcoin.Date = pd.to_datetime(bitcoin.Date)
bitcoin = bitcoin.sort_index(ascending=False)
bitcoin = bitcoin.reset_index()
bitcoin = bitcoin.drop(columns=['index'])

bitcoin_copy = bitcoin.copy(deep=True)
bitcoin_copy.rename(columns={'Date':'ds','price_bitcoin':'y'}, inplace=True)

date_y = bitcoin_copy[["ds","y"]]

if model_select == "fb-Prophet":
    st.header("FaceBook Prophet")
    st.markdown("""This is a model that takes the seasonality(yearly,weekly and daily) into consideration.
    It is a Supervised Model that attempts to predict the future depending on the past experiences.
    It detects change points from which the price suddenly changes from the general trend.""")

    m = Prophet()
    m.fit(date_y)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    fig1 = m.plot(forecast)

    st.pyplot(fig1)
    st.markdown("""The black dots show the daily price of bitcoin up until April 14, 2021, while the blue
    region shows the model's prediction of the price with a confidence level of 95%.
    This Model shows us that if the price was to grow organically, $50,000 USD per 
    Bitcoin as a price should have been achieved in 2022, therefore WE ARE IN A BUBBLE!.""")

elif model_select == 'LPPLS':
    st.header("LPPLS")
    st.markdown("""This modelprovides a flexible framework to detect bubbles and predict regime changes of a 
    financial asset. Anytime the price rises above the blue line, it is over-valued i.e. in a bubble and
    everytime it is below the blue line it's undervalued.""")
    time= np.linspace(0, len(bitcoin)-1, len(bitcoin))
    price = np.log(bitcoin['price_bitcoin'].values)
    observations = np.array([time, price])
    MAX_SEARCHES = 25
    lppls_model = lppls.LPPLS(observations=observations)
    tc, m, w, a, b, c, c1, c2 = lppls_model.fit(observations, MAX_SEARCHES, minimizer='Nelder-Mead')
    fig2 = lppls_model.plot_fit()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig2)

    st.markdown("""Since the model works on a linear time scale, with day 0 being January 2012, when our data
    starts from, we can see that as of April 2021, WE ARE IN A BUBBLE!.""")

st.header("Interested In Tweets?")
st.markdown("You can check if a Tweet is a positive one or a negative one about Bitcoin")

user_input = st.text_input("Enter Tweet Related to Bitcoin")

if user_input:

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    tweets = pd.read_csv('Merged Twitter Data.csv')
    tweets = tweets[["text","sentiment"]]

    def int_to_string(sentiment):
        if sentiment == 'positive':
            return 0
        elif sentiment == 'negative':
            return 1
        else:
            return 1
    tweets.sentiment = tweets.sentiment.apply(int_to_string)
    tweets.sentiment = pd.to_numeric(tweets.sentiment)
    tweets = tweets.dropna()

    train, test = train_test_split(tweets, test_size = 0.2, stratify = tweets['sentiment'], random_state=21)

    pipeline = Pipeline(steps= [('tfidf', TfidfVectorizer(lowercase=True,
                                                        max_features=1000,
                                                        stop_words= ENGLISH_STOP_WORDS)),
                                ('model', LogisticRegression())])

    # fit the pipeline model with the training data                            
    pipeline.fit(train.text, train.sentiment)
    user_inputs = []
    user_inputs.append(user_input)
    feeling = pipeline.predict(user_inputs)
    if feeling[0] == 0:
        st.write("It is a Positive Statement")
    
    if feeling[0] == 1:
        st.write('It is a Negative Statement')