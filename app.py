import streamlit as st
import pandas as pd
from PIL import Image
st.title(':blue[Call Center Analytics Dashboard] :telephone_receiver:')


df = pd.read_csv('processed_data.csv')
df['PAIN_POINTS'] = df['PAIN_POINTS'].replace('.','')
df['PAIN_POINTS'] = df['PAIN_POINTS'].str.lower()

col1, col2, col3, col4, col5 = st.columns(5)
total_cost = round((df['INPUT_PRICE_GPT3.5'].sum()*3) + df['OUTPUT_PRICE_SENTIMENT'].sum() + df['OUTPUT_PRICE_FOLLOW_UP'].sum(),3) 
col1.metric("Dashboard Cost","$"+str(total_cost), "$0.013")
col2.metric("Positivity Score", str(round(df['SENTIMENT_3_5'].mean(),2))+"/1.0", "-.08")
col3.metric("Resolved %", "86%", "4%")
col4.metric("Average Call Duration", "2m 37s", "49 s")
col5.metric("Total Calls", "23", "4")
col1, col2 = st.columns(2)


from wordcloud import WordCloud
import matplotlib.pyplot as plt
items = df['PAIN_POINTS'].to_list()
wordcloud = WordCloud(background_color='black', max_words=100)
wordcloud.generate(' '.join(items))
plt.rcParams['savefig.facecolor']='black'
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis('off') 
plt.savefig('WC.jpg') 
img= Image.open("WC.jpg") 
col1.write('Key Customer Pain Points')
col1.image(img)

df_b = pd.read_csv('monthly_calls.csv')
col2.write('Total Calls Per Month')
col2.line_chart(data=df_b, x='Month', y='Calls')
# change to col1
