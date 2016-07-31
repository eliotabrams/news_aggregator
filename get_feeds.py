#######################
###    Get Feeds    ###
#######################
# Description: Code to get RSS feeds from slanted news sources
# Authors: Eliot Abrams and Annie Liang


#######################
###      Setup      ###
#######################
import feedparser
import time
import os
import pandas as pd


#######################
###    Functions    ###
#######################
# Reads from hard coded RSS feeds that Annie picked
def get_feeds():
    news_table = pd.DataFrame(columns=['timestamp', 'title', 'summary', 'url', 'slant'])
    feeds = ['http://thinkprogress.org/world/issue/feed/',
             'http://fulltextrssfeed.com/front.moveon.org/feed',
             'http://feeds.bbci.co.uk/news/rss.xml',
             'http://www.glennbeck.com/feed/',
             'http://www.thestate.com/news/politics-government/?widgetName=rssfeed&widgetContentId=712015&getXmlFeed=true',
             'http://feeds.feedburner.com/Realclearpolitics-Articles'
            ]
    for feed in feeds:
        d = feedparser.parse(feed)
        for post in d.entries:
            slant = get_slant(feed)
            news_table.loc[len(news_table)] = [post.published,
                                               post.title,
                                               post.summary,
                                               post.link,
                                               slant ]
    return news_table

# Label the slant (will be updated later to label based on source AND text)
def get_slant(feed):
    feeds = ['http://thinkprogress.org/world/issue/feed/',
             'http://fulltextrssfeed.com/front.moveon.org/feed',
             'http://feeds.bbci.co.uk/news/rss.xml',
             'http://www.glennbeck.com/feed/',
             'http://www.thestate.com/news/politics-government/?widgetName=rssfeed&widgetContentId=712015&getXmlFeed=true',
             'http://feeds.feedburner.com/Realclearpolitics-Articles'
            ]
    slants = [0.06,0.19,0.22,0.98,0.94,0.93]
    return slants[feeds.index(feed)]    

# Save to CSV can easily modify
def save_feeds():
    destination = 'news_table.csv'
    if not (os.path.exists(destination)):
        print('creating new file')
        get_feeds().to_csv(destination, encoding='utf-8')
    else:
        print('appending to existing file')
        news_table = get_feeds().append(pd.read_csv(destination, encoding='utf-8'))
        news_table.drop_duplicates('title').to_csv(destination, encoding='utf-8')
    print('Done')


#######################
###       Main      ###
#######################
os.chdir('/Users/eliotabrams/Desktop/news_aggregator')
schedule.every(20).minutes.do(save_feeds())
while True:
    schedule.run_pending()
    time.sleep(1)

