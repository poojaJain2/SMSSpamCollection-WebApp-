import streamlit as st

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud,STOPWORDS

#dataset_loc = "SMSSpamCollection"
image_loc = "spamSms.jpg"
# sidebar
def load_sidebar():
    st.sidebar.subheader("SMSSpamCollection")
    st.sidebar.success("This is a UK forum in which cell phone users make public claims about SMS spam messages, most of them without reporting the very spam message received")
    st.sidebar.info("This corpus has been collected from free or free for research sources at the Internet:")
    st.sidebar.warning("Exploratory analysis of Spam messages :sunglasses:")

def load_data():
    df = pd.read_csv('SMSSpamCollection',sep="\t",names=['target','message'])
   # df = df.loc[:, ['airline_sentiment', 'airline', 'text']]
    return df

def load_description(df):
    
        # Preview of the dataset
        st.header("Data Preview")
        preview = st.radio("Choose Head/Tail?", ("Top", "Bottom"))
        if(preview == "Top"):
            st.write(df.head())
        if(preview == "Bottom"):
            st.write(df.tail())

        # display the whole dataset
        if(st.checkbox("Show complete Dataset")):
            st.write(df)

        # Show shape
        if(st.checkbox("Display the shape")):
            st.write(df.shape)
            dim = st.radio("Rows/Columns?", ("Rows", "Columns"))
            if(dim == "Rows"):
                st.write("Number of Rows", df.shape[0])
            if(dim == "Columns"):
                st.write("Number of Columns", df.shape[1])
        
        # show columns
        if(st.checkbox("Show the Columns")):
            st.write(df.columns)

# WordCloud
def load_wordcloud(df,kind):
        if(kind=='spam'):
            temp_df1 = df.loc[df['target']=='spam', :]
            words = ' '.join(temp_df1['message'])
            cleaned_word = " ".join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
            wc = WordCloud(stopwords=STOPWORDS, background_color='black', width=1600, height=800).generate(cleaned_word)
            wc.to_file("wc.png")
        

        if(kind=='ham'):
            temp_df2 = df.loc[df['target']=='ham', :]
            words = ' '.join(temp_df2['message'])
            cleaned_word = " ".join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
            wc = WordCloud(stopwords=STOPWORDS, background_color='black', width=1600, height=800).generate(cleaned_word)
            wc.to_file("wc.png")
            

def load_viz(df):
        st.header("Data visualization")
        st.subheader("Seaborn - SMSspamCollection Analysis")
        st.subheader("1.)")
        st.write(sns.countplot(x='target', data=df))
        st.pyplot()

       # ************

        st.subheader("2.)")
        st.write(sns.countplot(x='target', hue='target', data=df))
        st.pyplot()

        st.subheader("Wordcloud")
        type = st.radio("Choose the Sentiments:-" , ("Spam" , "ham"))
        load_wordcloud(df,type)
        st.image("wc.png", use_column_width = True)


def main():

    # sidebar
    load_sidebar()

    # Title/ text
    st.title('SMSSpamCollection')
    st.image(image_loc,use_column_width = True)
    st.text('Analyze how travelers in February 2015 expressed their feelings on Twitter')

    # loading the data
    df = load_data()

    # display description
    load_description(df)

    # data viz
    load_viz(df)



if(__name__ == '__main__'):
    main()
