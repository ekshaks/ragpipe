from typing import List
from .docnode import DocNode

def tokenize_remove_stopwords(text: str) -> List[str]:
    from nltk.stem import PorterStemmer
    # lowercase and stem words
    text = text.lower()
    stemmer = PorterStemmer()
    words = text.split(' ') 
    return [stemmer.stem(word) for word in words]

#https://github.com/dorianbrown/rank_bm25
from rank_bm25 import BM25Okapi


class BM25:
    def __init__(self, doc_list) -> None: #expect: doc.get_content() defined
        self.doc_list = doc_list
        self._corpus = [self._tokenizer(doc.get_content()) for doc in doc_list]
        self.bm25 = BM25Okapi(self._corpus)
        
    def _tokenizer(self, text):
        return tokenize_remove_stopwords(text)
    
    def retrieve(self, query, top_k=5):
        tokenized_query = self._tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        node_scores = [DocNode(li_node=x[0], score=x[1]) for x in zip(self.doc_list, scores)]
        nodes_scores_sorted = sorted(node_scores, key=lambda x: x.score or 0.0, reverse=True)
        return nodes_scores_sorted[: top_k]