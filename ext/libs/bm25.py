from typing import List

def tokenize_remove_stopwords(text: str) -> List[str]:
    from nltk.stem import PorterStemmer
    # lowercase and stem words
    text = text.lower()
    stemmer = PorterStemmer()
    words = text.split(' ') 
    return [stemmer.stem(word) for word in words]

def tokenize_remove_stopwords(text: str) -> List[str]:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    #nltk.download('punkt')
    #nltk.download('stopwords')

    tokens = word_tokenize(text.lower())  
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]  # Remove stopwords
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]  # Stemming

    return stemmed_tokens

#https://github.com/dorianbrown/rank_bm25
from rank_bm25 import BM25Okapi
import numpy as np

#from ragpipe.index import BaseIndex, DEFAULT_LIMIT
from ragpipe.docnode import ScoreNode

class RankBM25Index:
    def __init__(self, doc_list, doc_paths) -> None: #expect: doc.get_content() defined
        self.doc_list = doc_list
        self.doc_paths = doc_paths #
        self._corpus = [self._tokenizer(doc) for doc in doc_list]
        self.bm25 = BM25Okapi(self._corpus)
        
    def _tokenizer(self, text):
        return tokenize_remove_stopwords(text)
    
    def retrieve(self, query, limit=10):
        if isinstance(query, str):
            tokenized_query = self._tokenizer(query)
        else:
            assert isinstance(query, list)
            tokenized_query = query
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:limit]
        node_scores = [ScoreNode(li_node=self.doc_list[i], score=scores[i], doc_path=self.doc_paths[i]) #doc_path_index=i) 
                       for i in top_n]
        nodes_scores_sorted = sorted(node_scores, key=lambda x: x.score or 0.0, reverse=True)
        return nodes_scores_sorted[: limit]