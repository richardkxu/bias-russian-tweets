# Unveiling and Analyzing Bias in the Russian Troll Tweets Dataset
### Uvic Data Models 501 - Group Project 2 (Streamline Version)

## Prerequisites
Ensure the following Python libraries are installed:
- `pandas`
- `nltk`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `gensim`

Additionally, download the following NLTK data:
- `punkt`
- `stopwords`
- `wordnet`

## Files
- `common_word_list.txt`: Contains a list of common words found in all four subsets (Left_Troll, Right_Troll, NewsFeed, HashtagGamer) for analysis.
- CSV files: `LeftTroll.csv`, `RightTroll.csv`, `HashtagGamer.csv`, `NewsFeed.csv` for each subset.
- `GoogleNews-vectors-negative300.bin`: Google's pre-trained Word2Vec model. Download from [Google Drive](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g).

## Data Cleaning and Preprocessing
The script preprocesses tweets from each `selected_subset.csv`. This includes removing URLs, tokenization, stop words removal, and applying stemming and lemmatization. This will generate one `.csv` file for each subset, located in the `./preprocessed-data` directory.

1. Place all `.csv` files in the current directory.
2. Run the script: `python preprocess.py`.

## Train Word Embedding and Visualize
**Word2Vec Model Training**: Trains a custom Word2Vec model on preprocessed tweets.

**Vector Extraction**: Extracts word vectors for selected common words from both the custom model and Google's pre-trained model.

**Spectral Clustering**: Applies Spectral Clustering to the word vectors.

**Visualization**: Uses `matplotlib` to visualize the word clusters with annotations.

**Comparison**: Compares clustering results between the custom trained model and Google's pre-trained model.

1. make sure `Project2.py`, `common_word_list.txt`, `Selected_subset.csv` are in one directory.
2. Run the script: `python Project2.py`.

# Results
The script generates two types of visualizations:
1. Clusters from the custom Word2Vec model.
2. Clusters from Google's pre-trained Word2Vec model.
