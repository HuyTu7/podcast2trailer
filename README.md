# Podcast to Trailer 

For more traditional machine learning approaches, LDA and PageRank are good candidates. These algorithms approach the problem with finding the most important sentence(s) within the transcription. Essentially, we doing summarization for now. PageRank is more computationally costly for both space and time in comparison to LDA. The summary of both approaches are provided below:

- Latent Dirichlet Allocation (LDA): This is a generative probabilistic model that tries to discover the latent topics that are present in a corpus. 
    - Time Complexity: O(NIK) where N is the number of documents/sentences, I is the number of iterations, and K is the number of topics.
    - Space Complexity: O(N\*K) 

- PageRank: uses a graph-based approach to identify the most important sentences in a text. This is usually used in rank web pages in search engines. 
    - Time Complexity: O(N^3) where N is the number of sentences.
    - Space Complexity: O(N^2) 

For both approaches, we tend to observe a movie trailer usually 1.5 minutes out of 90 minutes movie, which approximates 1/60 ratio. For each transcription data point, we want to utilize this ratio to calculate the number of summarized sentences we need. 

### Example

![](https://github.com/HuyTu7/podcast2trailer/blob/main/recording.gif)


###  Repository Structure
```
podcast2trailer/
├── README.md
├── main.py
├── data_preprocessing.py
├── exploration.ipynb
```
* `README.md` - the description of podcast2trailer exercise description and approach. 

* `data_preprocessing.py` - loading the data and run preprocessing steps (remove stop words, non-alphabetic tokens, etc) on the data 

* `main.py` - implementation of two approaches (LDA and PageRank) end-to-end

* `exploration.ipynb` - playbook with some results analysis and for saving the summarized  


### Requirements
- Python 3
- pandas
- numpy
- nltk
- networkx
- sklearn
- gensim
