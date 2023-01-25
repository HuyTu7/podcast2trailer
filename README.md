# Podcast to Trailer 

For more traditional machine learning approaches, LDA and PageRank are good candidates. PageRank is more computationally costly for both space and time in comparison to LDA. The summary of both approaches are provided below:

- Latent Dirichlet Allocation (LDA): This is a generative probabilistic model that tries to discover the latent topics that are present in a corpus. 
    - Time Complexity: O(NIK) where N is the number of documents/sentences, I is the number of iterations, and K is the number of topics.
    - Space Complexity: O(N\*K) 

- PageRank: uses a graph-based approach to identify the most important sentences in a text. This is usually used in rank web pages in search engines. 
    - Time Complexity: O(N^3) where N is the number of sentences.
    - Space Complexity: O(N^2) 
    
###  Repository Structure
```
podcast2trailer/
├── README.md
├── main.py
├── data_preprocessing.py
```
* `README.md` - the description of podcast2trailer exercise description and approach. 

* `preprocessing.py` - 

* `main.py` - 


### Requirements
- Python 3
- pandas
- numpy
- nltk
- networkx
- sklearn
