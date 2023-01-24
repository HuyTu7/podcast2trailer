from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from data_preprocessing import *

MYPATH = "datasets/"
RATIO = 1 / 60  # a movie trailer usually 1.5 mins out of 90 mins movie (~ 1/60 ratio)


def pagerank_summarize(embeddings: Dict, transcript: str):
    sentences = sent_tokenize(transcript)
    clean_sentences = preprocessing(sentences)
    sentence_vectors = []
    # make sentences vector
    for i in clean_sentences:
        if len(i) != 0 and i.split():
            v = sum([embeddings.get(w, np.zeros((300,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            v = np.zeros((300,))
        sentence_vectors.append(v)
    sim_mat = np.zeros([len(sentences), len(sentences)])

    # make similarity matrix
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = \
                    cosine_similarity(sentence_vectors[i].reshape(1, 300), sentence_vectors[j].reshape(1, 300))[0, 0]
    nx_graph = nx.from_numpy_array(sim_mat)
    pagerank_scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((pagerank_scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    num_of_sent = round(RATIO * len(sentences))
    summary = " ".join([ranked_sentences[i][1] for i in range(num_of_sent)])
    return summary


if __name__ == '__main__':
    word_embeddings = load_word_embeddings()
    examples = read_data(MYPATH)
    for fname, transcription in examples.items():
        print("ORIGINAL: ")
        print(transcription)
        print()
        print("SUMMARY: ")
        print(pagerank_summarize(word_embeddings, transcription))
