from flask import Flask, request, render_template, session, redirect
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
import spacy
from spacy import displacy
from collections import Counter
# from IPython.core.display import display, HTML
import glob
import codecs
import multiprocessing
import os
import re
import gensim.models.word2vec as w2v
import sklearn.manifold
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import seaborn as sns
from textblob.classifiers import NaiveBayesClassifier as NBC
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk import FreqDist
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
from pathlib import Path
from afinn import Afinn
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

plt.figure(figsize=(14,4))
af = Afinn()
sns.set(rc={'figure.figsize':(14,4)})
nltk.download("punkt")
nltk.download("stopwords")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stop = stopwords.words('english')
nlp = spacy.load('en')

app = Flask(__name__)

#-----------------------------------------------------------------------------------------------------

#ner
def preprocess(sent):
	sent = nltk.word_tokenize(sent)
	sent = nltk.pos_tag(sent)
	return sent

#Sentence Similarity
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

#Sentiment Analysis
def visualise_sentiments(data):
	plt.axis('off')
	plt.close()
	return sns.heatmap(pd.DataFrame(data).set_index("Sentence").T,center=0, annot=True, cmap = "PiYG")

#text analysis
def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))

#topic modeling
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop])
    return rev_new

def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
    output = []
    for sent in texts:
        doc = nlp(u" ".join(sent)) 
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output

def freq_words(x, terms = 15):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = terms) 
    plt.figure(figsize=(14,4))
    axe = sns.barplot(data=d, x= "word", y = "count")
    axe.set(ylabel = 'Count')
    return axe


def fig_to_base64(fig):
	img = io.BytesIO()	
	# ax.figure.savefig('file.png')
	fig.figure.savefig(img, format='png',
	            bbox_inches='tight')
	img.seek(0)

	return base64.b64encode(img.getvalue())

#-----------------------------------------------------------------------------------------------------------

#Flask Handlers

@app.route('/')
def index():
	df = pd.read_csv('static/data/data.csv')
	df['feedback'] = df['feedback'].str.replace("[^a-zA-Z#]", " ")
	stop_words = stopwords.words('english')
	df['feedback'] = df['feedback'].apply(lambda x: u' '.join([w for w in x.split() if len(w)>2]))
	# remove stopwords from the text
	feedback = [remove_stopwords(r.split()) for r in df['feedback']]
	# make entire text lowercase
	feedback = [r.lower() for r in feedback]
	tokenized_feedback = pd.Series(feedback).apply(lambda x: x.split())
	feedback_2 = lemmatization(tokenized_feedback)
	feedback_3 = []
	for i in range(len(feedback_2)):
	    feedback_3.append(u' '.join(feedback_2[i]))

	df['feedback'] = feedback_3
	freq = freq_words(df['feedback'], 15)
	encoded = fig_to_base64(freq)
	freq1 = "data:image/png;base64, {}".format(encoded.decode('utf-8'))
	plt.close('all')
	plt.clf()
	return render_template('index.html', freq=freq1)

@app.route('/tables')
def html_table():
	dfun = pd.read_csv('static/data/data.csv')
	return render_template('tables.html', raw=dfun)

@app.route('/ner')
def ner():
	dfun = pd.read_csv('static/data/data.csv')
	sent = preprocess(dfun['feedback'][0])
	dfunpos = pd.DataFrame(sent, columns=['Words', 'Part of Speech Tags'])
	pattern = 'NP: {<DT>?<JJ>*<NN>}'
	cp = nltk.RegexpParser(pattern)
	cs = cp.parse(sent)
	iob_tagged = tree2conlltags(cs)
	dfuniob = pd.DataFrame(iob_tagged, columns=['Word', 'POS', 'IOB-Chunk'])
	# snt = unicode(dfun['feedback'][0])
	# doc = nlp(snt)
	# svg = displacy.render(doc, style="dep")
	# file_name = '-'.join([w.text for w in doc if not w.is_punct]) + ".svg"
	# output_path = Path("/home/madhav/Downloads/AIB/displaCyimages/" + file_name)
	# output_path.open("w", encoding="utf-8").write(svg)
	plt.close('all')
	return render_template('chartjs.html', pos=dfunpos, iob=dfuniob)

@app.route('/ss')
def ss():
	f = open("static/data/data.txt", "r")
	a = f.read()
	f.close()
	raw_sentences = tokenizer.tokenize(a)
	sentences = []
	for raw_sentence in raw_sentences:
	    if len(raw_sentence) > 0:
	        sentences.append(sentence_to_wordlist(raw_sentence))
	token_count = sum([len(sentence) for sentence in sentences])
	num_features = 300
	min_word_count = 3
	num_workers = multiprocessing.cpu_count()
	context_size = 7
	downsampling = 1e-3
	seed = 1
	thrones2vec = w2v.Word2Vec(sg=1, seed=seed, workers=num_workers, size=num_features, min_count=min_word_count, window=context_size, sample=downsampling)
	thrones2vec.build_vocab(sentences)
	corpus_count = thrones2vec.wv.vocab
	thrones2vec.train(sentences, total_examples=corpus_count, epochs=100)
	if not os.path.exists("trained"):
		os.makedirs("trained")
	thrones2vec.save(os.path.join("trained", "thrones2vec.w2v"))
	thrones2vec = w2v.Word2Vec.load(os.path.join("trained", "thrones2vec.w2v"))
	tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
	all_word_vectors_matrix = thrones2vec.wv.vectors
	all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
	points = pd.DataFrame(
	[
	    (word, coords[0], coords[1])
	    for word, coords in [
	        (word, all_word_vectors_matrix_2d[thrones2vec.wv.vocab[word].index])
	        for word in thrones2vec.wv.vocab
	    ]
	],
	columns=["word", "x", "y"]
	)
	sigword = points.head(10)
	#Preplot
	# sns.set_context("poster")
	scat = points.plot.scatter("x", "y", s=10, figsize=(14, 4))
	encoded = fig_to_base64(scat)
	scat1 = "data:image/png;base64, {}".format(encoded.decode('utf-8'))
	def plot_region(x_bounds, y_bounds):
		slice = points[
		    (x_bounds[0] <= points.x) &
		    (points.x <= x_bounds[1]) & 
		    (y_bounds[0] <= points.y) &
		    (points.y <= y_bounds[1])
		]

		axe = slice.plot.scatter("x", "y", s=35, figsize=(14, 4))
		for i, point in slice.iterrows():
		    axe.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)
		return axe
	focus = plot_region(x_bounds=(0, 10), y_bounds=(0, 10))
	encodedfoc = fig_to_base64(focus)
	focus1 = "data:image/png;base64, {}".format(encodedfoc.decode('utf-8'))
	#Postplot
	data = pd.DataFrame(thrones2vec.wv.most_similar("data"), columns=['Most Similar Term', 'Score'])
	figure = pd.DataFrame(thrones2vec.wv.most_similar("figure"), columns=['Most Similar Term', 'Score'])
	issues = pd.DataFrame(thrones2vec.wv.most_similar("portfolio"), columns=['Most Similar Term', 'Score'])
	def nearest_similarity_cosmul(start1, end1, end2):
	    similarities = thrones2vec.wv.most_similar_cosmul(
	        positive=[end2, start1],
	        negative=[end1]
	    )
	    start2 = similarities[0][0]
	    # print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
	    return "{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals())
	one = nearest_similarity_cosmul("issues", "graph", "data")
	mgt = nearest_similarity_cosmul("all", "managed", "results")
	# return sigword, scat, focus, data, issues, one, mgt
	plt.close('all')
	plt.clf()
	return render_template('chartjs2.html', scat=scat1, focus=focus1, data=data, issues=issues)

@app.route('/sa')
def sentiment():
	dfsen = pd.read_csv('static/data/datapolsubsentiment.csv')
	sentence = dfsen['feedback'][7]

	sentimentviz = visualise_sentiments({
	      "Sentence":["SENTENCE"] + sentence.split(),
	      "Sentiment":[TextBlob(sentence).polarity] + [TextBlob(word).polarity for word in sentence.split()],
	      "Subjectivity":[TextBlob(sentence).subjectivity] + [TextBlob(word).subjectivity for word in sentence.split()]
	})

	encoded = fig_to_base64(sentimentviz)
	sv = "data:image/png;base64, {}".format(encoded.decode('utf-8'))

	df = pd.read_csv('static/data/AIB - Sentiment - Train.csv')
	df['sentiment_scores'] = [af.score(article) for article in df['feedback']]
	df['sentiment_category'] = ['positive' if score > 0 
	                          else 'negative' if score < 0 
	                              else 'neutral' 
	                                  for score in df['sentiment_scores']]
	df1 = pd.DataFrame([list(df['label']), df['sentiment_scores'], df['sentiment_category']]).T
	df1.columns = ['category', 'sentiment_score', 'sentiment_category']
	df1['sentiment_score'] = df1.sentiment_score.astype('float')
	dfout = df1.groupby(by=['category']).describe()

	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
	sp = sns.stripplot(x='category', y="sentiment_score", 
	                   hue='category', data=df1, ax=ax1)
	bp = sns.boxplot(x='category', y="sentiment_score", 
	                 hue='category', data=df1, palette="Set2", ax=ax2)
	t = f.suptitle('Visualizing Sentiment', fontsize=12)
	canvas = FigureCanvas(f)
	encodedvs = fig_to_base64(canvas)
	vs = "data:image/png;base64, {}".format(encodedvs.decode('utf-8'))
	fc = sns.factorplot(x="category", hue="sentiment_category", 
	                    data=df1, kind="count", 
	                    palette={"negative": "#FE2020", 
	                             "positive": "#BADD07", 
	                             "neutral": "#68BFF5"})

	figfc = fc.fig
	canvasfc = FigureCanvas(figfc)
	encodedfc = fig_to_base64(canvasfc)
	fcviz = "data:image/png;base64, {}".format(encodedfc.decode('utf-8'))
	pos_idx = df1[(df1.category=='Compliment') & (df1.sentiment_score == 5.0)].index[0]
	neg_idx = df1[(df1.category=='Remark') & (df1.sentiment_score == -4.0)].index[0]
	mostneg = df.iloc[neg_idx][['feedback']][0]
	mostpos = df.iloc[pos_idx][['feedback']][0]

	#Textblob
	dfnew = pd.read_csv('static/data/AIB - Sentiment - Train.csv')

	# compute sentiment scores (polarity) and labels
	dfnew['sentiment_scores_tb'] = [round(TextBlob(article).sentiment.polarity, 3) for article in dfnew['feedback']]
	dfnew['sentiment_category_tb'] = ['positive' if score > 0 
	                             else 'negative' if score < 0 
	                                 else 'neutral' 
	                                     for score in dfnew['sentiment_scores_tb']]


	# sentiment statistics per news category
	dfnew1 = pd.DataFrame([list(dfnew['label']), dfnew['sentiment_scores_tb'], dfnew['sentiment_category_tb']]).T
	dfnew1.columns = ['category', 'sentiment_score', 'sentiment_category']
	dfnew1['sentiment_score'] = dfnew1.sentiment_score.astype('float')
	dfout1 = dfnew1.groupby(by=['category']).describe()
	plt.close('all')
	# return sentence, sentimentviz, dfout, vizsent, fc, mostneg, mostpos, dfout1
	return render_template('morisjs.html', sentence=sentence, sent=sv, dfout=dfout, vizsent=vs, fc=fcviz, mostneg=mostneg, mostpos=mostpos, dfout1=dfout1)


@app.route('/ta')
def textanalysis():
	train = pd.read_csv('static/data/datapolsubsentiment.csv')
	train['word_count'] = train['feedback'].apply(lambda x: len(str(x).split(" ")))
	train['char_count'] = train['feedback'].str.len()
	train['avg_word'] = train['feedback'].apply(lambda x: avg_word(x))
	train['stopwords'] = train['feedback'].apply(lambda x: len([x for x in x.split() if x in stop]))
	train[['feedback','stopwords']].head()
	train['feedback'] = train['feedback'].apply(lambda x: " ".join(x.lower() for x in x.split()))
	train['feedback'] = train['feedback'].str.replace('[^\w\s]','')
	train['feedback'] = train['feedback'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
	train = train.drop(['Unnamed: 0'], axis=1)
	freq = pd.Series(' '.join(train['feedback']).split()).value_counts()[:10]
	tf1 = (train['feedback'][1:5]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
	tf1.columns = ['words','tf']
	for i,word in enumerate(tf1['words']):
	    tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['feedback'].str.contains(word)])))

	tf1['tfidf'] = tf1['tf'] * tf1['idf']
	# return train, tf1
	plt.close('all')
	plt.clf()
	return render_template('echarts.html', train=train[:30], tf1=tf1[:6])

	

@app.route('/tm')
def topicmodeling():
	df = pd.read_csv('static/data/data.csv')
	df['feedback'] = df['feedback'].str.replace("[^a-zA-Z#]", " ")
	stop_words = stopwords.words('english')
	df['feedback'] = df['feedback'].apply(lambda x: u' '.join([w for w in x.split() if len(w)>2]))
	# remove stopwords from the text
	feedback = [remove_stopwords(r.split()) for r in df['feedback']]
	# make entire text lowercase
	feedback = [r.lower() for r in feedback]
	tokenized_feedback = pd.Series(feedback).apply(lambda x: x.split())
	feedback_2 = lemmatization(tokenized_feedback)
	feedback_3 = []
	for i in range(len(feedback_2)):
	    feedback_3.append(u' '.join(feedback_2[i]))

	df['feedback'] = feedback_3
	freq = freq_words(df['feedback'], 15)
	encoded = fig_to_base64(freq)
	freq1 = "data:image/png;base64, {}".format(encoded.decode('utf-8'))

	# dictionary = corpora.Dictionary(feedback_2)
	# doc_term_matrix = [dictionary.doc2bow(rev) for rev in feedback_2]
	# LDA = gensim.models.ldamodel.LdaModel
	# lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=7, random_state=100, chunksize=1000, passes=50)
	# lda_model.print_topics()
	# vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
	# pyLDAvis.save_html(vis, 'lda.html')
	# return freq, vis
	plt.close('all')
	plt.clf()
	return render_template('other_charts.html', freq=freq1)

if __name__ == '__main__':
    app.run()
