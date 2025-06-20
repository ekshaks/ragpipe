from ragpipe.index import BaseIndex, DEFAULT_LIMIT
from ragpipe.docnode import PathNode
from pathlib import Path

from typing import List, Dict


# TODO: investigate how to make self.tokenizer work (instead of bm25.tokenize)

class BM25sIndex(BaseIndex):

    def __init__(self, **data):
        super().__init__(**data)
        self.lang = data.get('lang', 'en')
        self.lang_full = data.get('lang_full', 'english')
        self.stopwords= data.get('stopwords', 'en')
        self.index_dir= Path(data.get('index_dir',"/tmp/ragpipe/bm25s_indices/"))
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)

        try:
            import bm25s
            import Stemmer  # optional: for stemming
            from bm25s.tokenization import Tokenized

        except Exception as e:
            print('To use BM25s: please `pip install bm25s PyStemmer numba`')
            raise e

        self.all_doc_paths = []
        self.all_corpus_tokens = Tokenized(ids=[], vocab={})

        self.stemmer = Stemmer.Stemmer("english")
        self.tokenizer = bm25s.tokenization.Tokenizer(stemmer=self.stemmer, stopwords=self.lang)

        self.retriever_bm25s = bm25s.BM25()
        self.bm25s = bm25s
    
    def load(self):
        #assert False #for debuggin
        sc = self.index_config.storage_config
        assert sc is not None, 'BM25s: index_config.storage_config is None, cannot load.'
        collection_name = sc.collection_name
        self.coll_dir = self.index_dir / collection_name
        self.retriever_bm25s = self.bm25s.BM25.load(self.coll_dir, mmap=True) #, load_corpus=True) #loads corpus records
        self.tokenizer.load_vocab(self.coll_dir)

    def add(self, docset, is_query=False, docs_already_encoded=False, NUM_DOCS=100, truncate_text_at=None ):
        NUM_DOCS = 500
        docs = docset.items #[:NUM_DOCS]
        docs = [doc[:truncate_text_at] for doc in docs]
        doc_paths = docset.item_paths #[:NUM_DOCS]

        doc_list = docs # list of text
        self.all_doc_paths.extend(doc_paths)

        corpus_tokens = self.bm25s.tokenize(doc_list, stemmer=self.stemmer)
        print('add: tokenizer=', self.tokenizer)


        if is_query:
            self.doc_embeddings.extend(docs) #query is passthrough encoded
            self.doc_paths = doc_paths
            self.is_query = True
        else:
            self.retriever_bm25s = self.bm25s.BM25() #corpus=doc_records, backend="numba") #TODO: FIX! allows 'add' only once. accumulate all corpus records and build again?
            #self.retriever_bm25s.index(self.all_corpus_tokens)
            #print(corpus_tokens)
            self.retriever_bm25s.index(corpus_tokens)
            #self.all_doc_paths.extend(doc_paths)
        
        
        #print('BMXIndex.add index_config: ', self.index_config)
        sc = self.index_config.storage_config
        if sc is not None and not is_query:
            collection_name = sc.collection_name
            self.coll_dir = self.index_dir / collection_name
            self.coll_dir.mkdir(parents=True, exist_ok=True)

            self.retriever_bm25s.save(self.coll_dir)
            self.tokenizer.save_vocab(self.coll_dir)
            self.tokenizer.save_stopwords(self.coll_dir)
            print('BM25s: saving to coll name', collection_name, self.coll_dir)
        

    def retrieve(self, rep_query, limit=DEFAULT_LIMIT):
        print('retriever: tokenizer=', self.tokenizer)


        query_tokens = self.bm25s.tokenize(rep_query, stemmer=self.stemmer) #update_vocab=False
        #print('BM25s rep query: == ',  rep_query, query_tokens)
        results, scores = self.retriever_bm25s.retrieve(query_tokens, k=limit, corpus=self.all_doc_paths) # corpus=corpus_ids, filter=bitmask
        # print(results[0])
        # print(scores[0])
        # print(results.shape)
        # print(scores.shape)
        # #print('score sum', scores.sum())
        #documents = results[0]

        docnodes = []
        for i in range(results.shape[1]):
            doc = results[0, i]
            doc_path = doc #corpus field in .retrieve | alternatively add corpus to bm25 initialization
            #print(doc)
            score = scores[0, i]
            docnodes.append(PathNode(doc_path=doc_path, score=score, is_ref=True))
        #     doc, score = results[0, i], scores[0, i]
        #     print(f"Rank {i+1} (score: {score:.2f}): {doc}")

        return docnodes
    
    
    