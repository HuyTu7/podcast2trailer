import string
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim import corpora, models
import networkx as nx
from transformers import pipeline
from data_preprocessing import *

MYPATH = "datasets/"
RATIO = 1 / 60  # a movie trailer usually 1.5 mins out of 90 mins movie (~ 1/60 ratio)
TRANSFORMER_LIMIT = 512  # processing limit for the transformer model type


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


def transformer_summarizer(transcript: str) -> str:
    summarizer = pipeline("summarization")

    word_tokens = word_tokenize(transcript)
    total_N = len(word_tokens)
    num_bins = int(total_N / TRANSFORMER_LIMIT)
    left = 0
    tmp_summaries = []

    # break the transcript to multiple batch of sentences (paragraphs)
    # using window method of two cursors
    for i in range(num_bins):
        right = (i + 1) * TRANSFORMER_LIMIT if i + 1 < num_bins else total_N - 1

        # only split the string if the sentence completes.
        while word_tokens[right][-1] not in "?!.":
            right -= 1
            if word_tokens[right][-1] in "?!." and TRANSFORMER_LIMIT * 2 - (right - left) < 100:
                right -= 1
        text_len = right - left
        sub_string = "".join([" " + i if not i.startswith("'")
                                         and i not in string.punctuation else i
                              for i in word_tokens[left:right + 1]]).strip()

        # after trying 1/60, most of the summaries make no sense
        # so increase the limit for the returned tokens
        tmp_summaries.append(summarizer(sub_string, min_length=int(text_len / 45), max_length=int(text_len / 30)))
        left = right + 1

    summary = ". ".join([summary[0]['summary_text'] for summary in tmp_summaries])
    return summary
