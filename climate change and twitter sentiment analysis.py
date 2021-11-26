# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 22:33:42 2021

@author: fegoa
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 02:08:46 2021

@author: fegoa
"""

# Tweeter Scraping
# importing libraries and packages
import snscrape.modules.twitter as sntwitter
import pandas as pd

# Creating list to append scrapped tweet data 
tweets_list = []
#maximum iteration before terminating the search
maxiter=10000
# # Using TwitterSearchScraper module from snscrape to scrape tweet data from twitters 
#and appending the tweets to to tweets_list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('cop26 + climate change since:2021-10-31 until:2021-11-13').get_items()):
    if i>maxiter:
        break
    # appending the tweer data,ID,content,user ID, number of retweet,mentioned users, media, tweet language,number of likes to tweets_list
    tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username,tweet.retweetCount,tweet.mentionedUsers,tweet.media,tweet.lang,tweet.likeCount])
    
# Creating a dataframe from the tweets_list above
tweets_df2 = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'No of retweets','mentioned users','media', 'lang', 'No of likes'])


# import nltk library for sentiment analysis
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
# import regular expression library for cleaning the tweets
import re
nltk.download('words')
words = set(nltk.corpus.words.words())

# Function to clean the tweets by removing @, http links and #
def cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = " ".join(tweet.split())
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet)
          if w.lower() in words or not w.isalpha())
    return tweet
  
# applying the cleaner function to our scraped tweet and creating a new column tweet_clean  
tweets_df2['tweet_clean'] = tweets_df2['Text'].apply(cleaner)


list1 = []
# iterating through the cleaned tweet to its SentimentIntensityAnalyzer polartity score and appending to list 1
for i in tweets_df2['tweet_clean']:
    list1.append((sid.polarity_scores(str(i)))['compound'])
 # Converting List 1 to a pandas series and creating a sentiment column in our tweet data (tweets_df2)   
tweets_df2['sentiment'] = pd.Series(list1)

# defining a function to classify the sentiment based on the polarity score
def sentiment_category(sentiment):
    label = ''
    if(sentiment>0.05):
        label = 'positive'
    elif(-0.05<= sentiment<= 0):
        label = 'neutral'
    else:
        label = 'negative'
    return(label)
# apply the sentiment_category function to the sentiment column and creating a new sentiment category column
tweets_df2['sentiment_category'] = tweets_df2['sentiment'].apply(sentiment_category)
text= tweets_df2['sentiment']
# function to get the tweet hashtags
def getHashtags(tweet):
    tweet=tweet.lower()
    tweet=re.findall(r'\#\w+',tweet)
    return "".join(tweet)
# Applying the getHastags function to the text column of the tweets data (tweets_df2)
tweets_df2['Hashtags']=tweets_df2['Text'].apply(getHashtags)

# obtaining the number of unique tweets based on the username column
tweets_unique=tweets_df2['Username'].unique()

# Obtaining only english tweets
tweets_df2=tweets_df2[tweets_df2['lang']=='en']
# Saving the the English tweets to a CSV file
tweets_df2.to_csv('COP26_POST_modified.csv', index= False)

# Sorting the tweets in descending order based on the number of tweet likes
tweets_top_10=tweets_df2.sort_values('No of likes', ascending=False)
# saving the sorted tweets above to a csv file
tweets_top_10.to_csv('COP26_POST_TOP_10.csv',index=False)

# obtaining the overall sentiments
Overall_sentiment_average=tweets_df2['sentiment'].mean()

from nltk.tokenize import word_tokenize
import textblob
import matplotlib.pyplot as plt
import csv
from nltk import pos_tag
import nltk
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO # For emojis
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
import string
from nltk.corpus import stopwords, words

# defining a preprocessTweets function
def preprocessTweets(tweet):
    tweet = tweet.lower()  #has to be in place
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#|\d+', '', tweet)
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)  # convert string to tokens
    filtered_words = [w for w in tweet_tokens if w not in stop_words]
    filtered_words = [w for w in filtered_words if w not in emojis]
    filtered_words = [w for w in filtered_words if w in word_list]

    # Remove punctuations
    unpunctuated_words = [char for char in filtered_words if char not in string.punctuation]
    unpunctuated_words = ' '.join(unpunctuated_words)

    return "".join(unpunctuated_words)  # join words with a space in between them

# defining a getAdjectives function
def getAdjectives(tweet):
    tweet=word_tokenize(tweet)
    tweet=[word for (word,tag) in pos_tag(tweet) if tag=="JJ"]
    return " ".join (tweet)
# Defining my NLTK stop words and my user-defined stop words
stop_words = list(stopwords.words('english'))
user_stop_words = [ 'year', 'many', 'much', 'amp', 'next', 'cant', 'wont', 'hadnt',
                    'havent', 'hasnt', 'isnt', 'shouldnt', 'couldnt', 'wasnt', 'werent',
                    'mustnt', '’', '...', '..', '.', '.....', '....', 'been…', 'one', 'two',
                    'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'aht',
                    've', 'next','new']
alphabets = list(string.ascii_lowercase)
stop_words = stop_words + user_stop_words + alphabets
word_list = words.words()  # all words in English language
emojis = list(UNICODE_EMOJI.keys())  # full list of emojis

# apply the preprocessTweets function to the tweet data text column
tweets_df2['Preprocessed']=tweets_df2['Text'].apply(preprocessTweets)

# apply the getAdjectives function to the preprocessed column
tweets_df2['tweet_Adjectives']=tweets_df2['Preprocessed'].apply(getAdjectives)

import numpy as np  
from PIL import Image
tweets_long_string=tweets_df2['tweet_Adjectives'].tolist()
tweets_long_string=" ".join(tweets_long_string)
image=np.array(Image.open('Twitter.png'))

# Creating a wordcloud image of the most common words
from wordcloud import WordCloud
fig=plt.figure()  #initiate figure object
fig.set_figwidth(14)
fig.set_figheight(18)

plt.imshow(image, cmap=plt.cm.gray, interpolation='bilinear') #Display data as an image
plt.axis('off')
plt.show()

import random
# Function to create blue colour for the word cloud
def blue_color_func(word,font_size,position,orientation,random_state=35,**kwargs):
    return "hsl(100, 100%%, %d%%)" % random.randint(20,60)
    

twitter_wc= WordCloud(background_color='white',max_words=100,mask=image)
twitter_wc.generate(tweets_long_string)

fig=plt.figure()
fig.set_figwidth(14)
fig.set_figheight(18)

# Visualizing the wordcloud image
plt.imshow(twitter_wc.recolor(color_func=blue_color_func,random_state=3),interpolation='bilinear')
plt.axis('off')
plt.show()

import plotly.express as px # To make express plots in Plotly
import chart_studio.tools as cst # For exporting to Chart studio
import chart_studio.plotly as py # for exporting Plotly visualizations to Chart Studio
import plotly.offline as pyo # Set notebook mode to work in offline
pyo.init_notebook_mode()
import plotly.io as pio # Plotly renderer
import plotly.graph_objects as go # For plotting plotly graph objects
from plotly.subplots import make_subplots #to make more than one plot in Plotly

# Convert the tweet adjectives column to a list
tweets_long_string = tweets_df2['tweet_Adjectives'].tolist()
tweets_list=[]

for item in tweets_long_string:
    item = item.split()
    for i in item:
        tweets_list.append(i)
from collections import Counter
counts = Counter(tweets_list)
df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
df.columns = ['Words', 'Count']
df.sort_values(by='Count', ascending=False, inplace=True)

# print(px.colors.sequential.Blues_r) to get the colour list used here. Please note, I swatched some colours

# Define my colours for the Plotly Plot
colors = ['rgb(0,100,0)', 'rgb(0,128,0)', 'rgb(34,139,34)', 'rgb(46,139,87)',
            'rgb(60,179,113)', 'rgb(127,255,122)','rgb(144,238,144)', 
            'rgb(152,251,152)', 'rgb(224,255,240)', 'rgb(240,255,240)']

# Set layout for Plotly Subplots
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, { "type": "domain"}]],
                    vertical_spacing=0.001)

# Add First Plot
fig.add_trace(go.Bar(x = df['Count'].head(10), y=df['Words'].head(10),marker=dict(color='rgba(0,180,0, 1)',
            line=dict(color='Black'),),name='Bar Chart',orientation='h'), 1, 1)

# Add Second Plot
fig.add_trace(go.Pie(labels=df['Words'].head(10),values=df['Count'].head(15),textinfo='label+percent',
                    insidetextorientation='radial', marker=dict(colors=colors, line=dict(color='Green')),
                    name='Pie Chart'), 1, 2)
# customize layout
fig.update_layout(shapes=[dict(type="line",xref="paper", yref="paper", x0=0.5, y0=0, x1=0.5, y1=1.0,
         line_color='DarkSlateGrey', line_width=1)])

# customize plot title
fig.update_layout(showlegend=False, title=dict(text="COP26 Top Ten Most Tweeted Words",
                  font=dict(size=18, )))

# Customize backgroound, margins, axis, title
fig.update_layout(yaxis=dict(showgrid=False,
                             showline=False,
                             showticklabels=True,
                             domain=[0, 1],
                             categoryorder='total ascending',
                             title=dict(text='Common Words', font_size=14)),
                             xaxis=dict(zeroline=False,
                             showline=False,
                             showticklabels=True,
                             showgrid=True,
                             domain=[0, 0.42],
                             title=dict(text='Word Count', font_size=14)),
                             margin=dict(l=100, r=20, t=70, b=70),
                             paper_bgcolor='rgba(255,255,255,1)',
                             plot_bgcolor='rgba(255,255,255,1)')

# Specify X and Y values for Annotations
x = df['Count'].head(10).to_list()
y = df['Words'].head(10).to_list()

# Show annotations on plot
annotations = [dict(xref='x1', yref='y1', x=xa + 350, y=ya, text=str(xa), showarrow=False) for xa, ya in zip(x, y)]

fig.update_layout(annotations=annotations)
fig.show(renderer = 'png')



tweets_df2['sentiment_category'].value_counts()
bar_chart=tweets_df2['sentiment_category'].value_counts().rename_axis('sentiment_category').to_frame('Total_tweets').reset_index()
bar_chart
sentiments_barchart = px.bar(bar_chart, x = 'sentiment_category', y='Total_tweets', color='sentiment_category', color_discrete_map={'positive':'green','negative':'red','neutral':'grey'})

sentiments_barchart.update_layout(title='Sentiments Results',
                                  margin={"r": 0, "t": 30, "l": 0, "b": 0})

sentiments_barchart.show(renderer = 'png')
#py.plot(sentiments_barchart, filename = 'Sentiments Results', auto_open=True)



#py.plot(fig, filename = 'Twitter Users 2020 Refelections (10 Most Common Words)', auto_open=True)
        
        





