from ragpipe.index import BaseIndex, DEFAULT_LIMIT
from ragpipe.docnode import ScoreNode

class BMXIndex(BaseIndex):
    def __init__(self, **data):
        super().__init__(**data)
        try:
            from baguetter.indices import BMXSparseIndex
        except Exception as e:
            print('To use BMX: please `pip install baguetter`')
            raise e
        self.idx = BMXSparseIndex()

    def add(self, docs, doc_paths, is_query=False, docs_already_encoded=False):
        #doc_ids = [str(i) for i in range(len(docs))] 
        #print('BMX: ', len(doc_paths))
        if is_query:
            self.doc_embeddings.extend(docs) #query is passthrough encoded
        else:
            self.idx.add_many(doc_paths, docs, show_progress=True)

    def retrieve(self, rep_query, limit=DEFAULT_LIMIT):
        results = self.idx.search(rep_query) #already sorted
        #print('BMXIndex results: ', results)
        keys, scores = results.keys[:limit], results.scores[:limit]
        docnodes = [ScoreNode(doc_path=key, score=score, is_ref=True) for key, score in zip(keys, scores)]
        return docnodes
    
    
    