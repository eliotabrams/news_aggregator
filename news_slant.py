############################
###   News Sentiment     ###
############################
# Authors: Eliot Abrams


############################
###        Setup         ###
############################

# Packages
import json
import os
import spacy
import csv
import string 
import re
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import linear_model
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import classification_report
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
import nltk    
from nltk.stem.porter import PorterStemmer
from spacy.en import English

# Other
parser = English()
path = 'N:\\Projects\\NewsProjects\\NewsSocial\\Analysis\\tensorFlow\\Data'
bigramsPath = '%s\\%s' % (path, 'by_party')
n_samples = 2000
n_features = 1000
n_topics = 100
n_top_words = 20
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "“", "”", "'ve"]


############################
###      Functions       ###
############################

# Read in the last three months (as of 7/1ish) of scrapped articles from "far out" news sources
# The "far out" news sources are:
# 730 billoreilly.com
# 484 glennbeck.com
# 499 thinkprogress.org
# 587 alternet.org
def read_far_out_articles(path):
	container = '%s\\pubs_730_484_499_587_articles_8_1_2014_to_2_6_2015.json' % path
	articles = pd.DataFrame(json.loads(open(container).read()))
	articles['polbias'] = articles['publisher_id'].map(lambda x: 'conservative' if x in [730, 484] else 'liberal')
	return articles

# Read in 2400 articles labeled with liberal, conservative, neutral by mechanical turkers
def read_mech_turk_articles(path):
	dataPath = '%s\\%s' % (path, 'SlantData\\TextArticles')
	slantPath = '%s\\%s' % (path, 'SlantData\\polbias.csv')

	data = pd.DataFrame(columns=('id', 'text'))
	for filename in os.listdir(dataPath):
		filename = '%s\%s' % (dataPath, filename)
		text = open(filename).read().decode('utf8')
		articleId = re.sub("\D", "", filename)
		data.loc[len(data)] = [int(articleId), text]
	slant_data = pd.read_csv(slantPath)
	dataset = pd.merge(data, slant_data, how='left', on='id')
	return dataset

# Read in Matt and Jessie's congressional records bigrams
# and collapse the bigrams across all congresses
def read_party_bigrams(path):
	bigrams = pd.DataFrame()
	for filename in os.listdir(path):
		filename = '%s\%s' % (path, filename)
		bigrams = bigrams.append(pd.read_table(filename, sep='|', low_memory=False))
	bigrams = bigrams.groupby(['phrase','party']).sum().reset_index()
	bigrams = bigrams.pivot('phrase', 'party', 'phrasecount').reset_index()
	bigrams = bigrams[['phrase','D','R']]
	bigrams['R/D'] = bigrams['R'] / bigrams['D']
	bigrams['R-D'] = bigrams['R'] - bigrams['D']
	return bigrams[pd.notnull(bigrams['R/D'])]

# Not currently used, but could be used to clean articles with tweets
def cleanText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
    text = mentionFinder.sub("@MENTION", text)
    text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
    return text.lower()

# A custom function to tokenize the text using spaCy
# Does lemmatization, which I understand is better than stemming with the Portor algorithm generally
def tokenizeText(sample):
    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")
    return tokens

# A NLTK based tokenizer that does Porter stemming
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens, stemmer)
    return stems    

# Merges the bigrams from the congressional records onto text
def bigram_slant(text, bigrams):
	pairs = nltk.bigrams(tokenizeText(text))
	pairs = [' '.join(tup) for tup in pairs if not False in [False for wrd in tup if wrd in STOPLIST] ]
	pairs = [''.join(tup) for tup in pairs if not False in [False for wrd in tup if wrd in SYMBOLS] ]
	pairs_w_slant = pd.merge(pd.DataFrame(pairs, columns=['phrase']) , bigrams, how='left', on='phrase')
	return [pairs_w_slant['R/D'].mean(), 
			pairs_w_slant['R/D'].median(),
			pairs_w_slant['R/D'].max(),
			pairs_w_slant['R/D'].min(),
			pairs_w_slant['R-D'].mean(), 
			pairs_w_slant['R-D'].median(),
			pairs_w_slant['R-D'].max(),
			pairs_w_slant['R-D'].min()]

# Prints the top words for the topic modeling
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


############################
###        Main          ###
############################

#####  Create Labeled Data  #####
# Insheet the cleaned data
mech_turk_articles = read_mech_turk_articles(path)
far_out_articles = read_far_out_articles(path)

# Subset to select "higher quality" far out articles
# Currently selects 2400 articles of each slant using length to proxy for "high quality," which is not great
far_out_articles['text_length'] = far_out_articles['text'].map(lambda x: len(x))
conservative = far_out_articles.sort_values(by=['polbias', 'text_length'], ascending=[True, False])[500:2000]
liberal = far_out_articles.sort_values(by=['polbias', 'text_length'], ascending=[False, False])[500:2000]
dataset = mech_turk_articles.append(liberal).append(conservative).reset_index(drop=True)
dataset['bias'] = dataset['polbias'].map(lambda x: -1 if x == 'conservative' else 1 if x == 'liberal' else 0)


#####  Slant Analysis: TF-IDF and then Lasso Approach  #####
""" WHAT THROWS UP THE CONVERGENCE WARNING ??? """
# Do the TF-IDF 
tfidf_vectorizer = TfidfVectorizer(max_df=0.70, min_df=0.05, stop_words='english', ngram_range=(2,3))
train_vectors = tfidf_vectorizer.fit_transform(dataset['text'][2400:len(dataset)])
test_vectors = tfidf_vectorizer.transform(dataset['text'][0:2400])

# Classify
classifier = linear_model.LassoCV(fit_intercept=True, max_iter=500, selection='random')
classifier.fit(train_vectors, dataset['bias'][2400:len(dataset)])
coefs = pd.DataFrame( [tfidf_vectorizer.get_feature_names(), classifier.coef_.tolist()] ).T.sort_values(by=1)
results = pd.concat([dataset['bias'][0:2400], pd.Series(classifier.predict(test_vectors), name='predicted')], axis=1)
results['rounded'] = results['predicted'].map(lambda x: round(x))
pd.crosstab(results.bias, results.rounded, margins=True)


#####  Slant Analysis: Bigram Approach  #####
bigrams = read_party_bigrams(bigramsPath)
dataset['bigram_slant'] = dataset['text'].apply(lambda x: bigram_slant(x, bigrams))

# Multinomial
model = sm.MNLogit(dataset['bias'][0:2000].tolist(),dataset['bigram_slant'][0:2000].tolist())
results = model.fit()
results.summary()
results.predict(dataset['bigram_slant'][2000:2400].tolist()) # Returns the value of the CDF at the linear predictor

# Other (switching to Sklearn for ease of moving through different models)
clf = linear_model.SGDClassifier(fit_intercept=True, class_weight={0: 1, -1: 20, 1: 2})
clf = linear_model.LinearRegression()
clf = linear_model.LogisticRegression(fit_intercept=True,  solver='lbfgs', multi_class='multinomial')
result = clf.fit(dataset['bigram_slant'][0:2300].tolist(), dataset['bias'][0:2300].tolist())
comparison = pd.DataFrame([result.predict(dataset['bigram_slant'][2300:2400].tolist()),
			dataset['bias'][2300:2400].tolist()]).T
comparison.columns = ['predicted', 'actual']


#####  Topic Modeling  #####
# Because why not practice these things?
# Use tf-idf features for NMF
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(dataset[,1])

# Use tf (raw term count) features for LDA
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(dataset)

# Fit the NMF model
nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

# Fit the LDA model
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
lda.fit(tf)
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)



"""
In order to make the vectorizer => transformer => classifier easier to work with, scikit-learn provides a Pipeline class that behaves like a compound classifier:
>>>
>>> from sklearn.pipeline import Pipeline
>>> text_clf = Pipeline([('vect', CountVectorizer()),
...                      ('tfidf', TfidfTransformer()),
...                      ('clf', MultinomialNB()),
... ])
The names vect, tfidf and clf (classifier) are arbitrary. We shall see their use in the section on grid search, below. We can now train the model with a single command:
>>>
>>> text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

#classifier = linear_model.LogisticRegression(fit_intercept=True,  solver='lbfgs', multi_class='multinomial')
#classifier = linear_model.ElasticNet(l1_ratio=0.2)
"""