import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

file_path = 'common_word_list.txt' # Change to exact location used
with open(file_path, 'r') as file:
    word_list = file.readlines()

selected_words = []
for word in word_list:
    stripped_word = word.strip()
    selected_words.append(stripped_word)

tweets_df = pd.read_csv('./preprocessed-data/HashtagGamer.csv')  # Change to exact location used

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_tweet(tweet):
    if isinstance(tweet, str):
        # Removing URLs
        words = tweet.split()
        tweet_without_urls = []
        for word in words:
            if not word.startswith("http"):
                tweet_without_urls.append(word)
        tweet = ' '.join(tweet_without_urls)

        # Tokenization
        tokens = word_tokenize(tweet)

        # Stop Words Removal
        filtered_tokens = []
        for word in tokens:
            if word.lower() not in stop_words:
                filtered_tokens.append(word)
        tokens = filtered_tokens

        # Stemming and Lemmatization
        stemmed = []
        lemmatized = []
        for word in tokens:
            stemmed_word = porter.stem(word)
            stemmed.append(stemmed_word)

        lemmatized_word = lemmatizer.lemmatize(word)
        lemmatized.append(lemmatized_word)

        return {
            "original": tweet,
            "tokens": tokens,
            "stemmed": stemmed,
            "lemmatized": lemmatized
        }
    else:
        # Return None if tweet is not a string
        return None

# Preprocessing tweets
preprocessed_tweets = []
for tweet in tweets_df['content']:
    if isinstance(tweet, str):
        preprocessed_tweet = preprocess_tweet(tweet)
        preprocessed_tweets.append(preprocessed_tweet)

# Extracting tokenized tweets
tokenized_tweets = []
for tweet in preprocessed_tweets:
    if tweet is not None and len(tweet['lemmatized']) > 0:
        tokenized_tweets.append(tweet['lemmatized'])

model = Word2Vec(sentences=tokenized_tweets, vector_size=100, window=5, min_count=1, workers=4)

# Save the model to a file
# model.save("model_1.model")

# Load the model from the file, if already saved, to speed up the proccess
# model = Word2Vec.load("word2vec_model.model")

word_vectors = model.wv
selected_word_vectors = []
for word in selected_words:
    if word in word_vectors:
        vector = word_vectors[word]
        selected_word_vectors.append(vector)
selected_word_vectors = np.array(selected_word_vectors)

# Applying Spectral Clustering, assume using n_clusters=7, feel free to test out outher numbers -- @RZ
n_clusters = 7 
spectral_cluster = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=0)
clusters = spectral_cluster.fit_predict(selected_word_vectors)
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=0)
tsne_results = tsne.fit_transform(selected_word_vectors)
used_words = [word for word in selected_words if word in word_vectors]

# Plot clusters with annotation
plt.figure(figsize=(16, 10))
for i in range(7):
    cluster_indices = clusters == i
    plt.scatter(tsne_results[cluster_indices, 0], tsne_results[cluster_indices, 1], label=f'Cluster {i}')
    for word, xy in zip(np.array(list(used_words))[cluster_indices], tsne_results[cluster_indices]):
        plt.annotate(word, xy, xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

plt.legend()
plt.title("Custom Word2Vec Model Clusters")
plt.show()

####################################### Pre-trained Model ##############################################

from gensim.models import KeyedVectors

g_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
g_selected_words = [word for word in selected_words if word in g_model.key_to_index]
g_selected_word_vectors = np.array([g_model[word] for word in g_selected_words])
pca_g = PCA(n_components=50)
reduced_g_vectors = pca_g.fit_transform(g_selected_word_vectors)
clustering_g = SpectralClustering(n_clusters=7, assign_labels="discretize", random_state=0)
labels_g = clustering_g.fit_predict(reduced_g_vectors)
tsne_g_results = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=0).fit_transform(reduced_g_vectors)

# Plot clusters with annotation
plt.figure(figsize=(16, 10))
for i in range(7):
    cluster_indices = labels_g == i
    plt.scatter(tsne_g_results[cluster_indices, 0], tsne_g_results[cluster_indices, 1], label=f'Cluster {i}')
    for word, xy in zip(np.array(list(g_selected_words))[cluster_indices], tsne_g_results[cluster_indices]):
        plt.annotate(word, xy, xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

plt.legend()
plt.title("Google's Pre-trained Word2Vec Model Clusters")
plt.show()
