from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from gensim import corpora, models
import networkx as nx
from data_preprocessing import *

MYPATH = "datasets/"
RATIO = 1 / 60  # a movie trailer usually 1.5 mins out of 90 mins movie (~ 1/60 ratio)


def pagerank_summarizer(embeddings: Dict, transcript: str):
    sentences = sent_tokenize(transcript)
    clean_sentences = preprocessing(sentences)
    N = len(sentences)
    sentence_vectors = []

    # Build each sentence vector from words embedding
    for i in clean_sentences:
        if len(i) != 0 and i.split():
            v = sum([embeddings.get(w, np.zeros((300,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            v = np.zeros((300,))
        sentence_vectors.append(v)
    sim_mat = np.zeros([N, N])

    # Make sentences similarity matrix
    for i in range(N):
        for j in range(i + 1, N):
            sim_mat[i][j] = \
                cosine_similarity(sentence_vectors[i].reshape(1, 300), sentence_vectors[j].reshape(1, 300))[0, 0]

    # Apply PageRank algorithm
    nx_graph = nx.from_numpy_array(sim_mat)
    pagerank_scores = nx.pagerank(nx_graph)

    # Sort the sentences based on their textrank scores
    ranked_sentences = sorted(((pagerank_scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    num_of_sent = round(RATIO * N)
    summary = " ".join([ranked_sentences[i][1] for i in range(num_of_sent)])
    return summary


def lda_summarizer(transcript: str) -> str:
    sentences = sent_tokenize(transcript)
    clean_sentences = preprocessing(sentences)
    lda_topics = round(RATIO * len(sentences))

    # Tokenize sentences into words
    texts = [[word for word in sentence.lower().split()] for sentence in clean_sentences]

    # Create a dictionary of words
    dictionary = corpora.Dictionary(texts)

    # Create a bag of words representation of the text
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train the LDA model
    lda = models.LdaModel(corpus, num_topics=lda_topics, id2word=dictionary)

    # Get the weight of each topic for each sentence
    topic_weights = [lda[dictionary.doc2bow(text)] for text in texts]

    # Find the most important sentence for each topic
    important_sentences = []
    for topic_id in range(lda_topics):
        most_important_sentence = ""
        highest_weight = -1
        for i, weights in enumerate(topic_weights):
            for topic, weight in weights:
                if topic == topic_id and weight > highest_weight:
                    most_important_sentence = sentences[i]
                    highest_weight = weight
        important_sentences.append(most_important_sentence)

    summary = " ".join([important_sentences[i] for i in range(lda_topics)])
    return summary


if __name__ == '__main__':
    word_embeddings = load_word_embeddings()
    examples = read_data(MYPATH)
    for fname, transcription in examples.items():
        print("ORIGINAL: ")
        print(transcription)
        print()
        print("SUMMARY: ")
        print(pagerank_summarizer(word_embeddings, transcription))
