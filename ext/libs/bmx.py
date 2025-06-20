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
    
    def load(self):
        sc = self.index_config.storage_config
        assert sc is not None, 'BMX: index_config.storage_config is None, cannot load.'
        collection_name = sc.collection_name
        print(f'---- Loading BMXIndex...{collection_name}')
        self.idx = self.idx.load(collection_name)

    def add(self, docset, is_query=False, docs_already_encoded=False):
        docs = docset.items
        doc_paths = docset.item_paths
        #doc_ids = [str(i) for i in range(len(docs))] 
        #print('BMX: ', len(doc_paths))
        if is_query:
            self.doc_embeddings.extend(docs) #query is passthrough encoded
            self.doc_paths = doc_paths
            self.is_query = True
        else:
            self.idx.add_many(doc_paths, docs, show_progress=True)
        
        #print('BMXIndex.add index_config: ', self.index_config)
        sc = self.index_config.storage_config
        if sc is not None:
            assert not is_query
            collection_name = sc.collection_name
            print('BMX: saving to coll name', collection_name)
            print('id tokens: ', len(self.idx.key_mapping))
            self.idx.save(collection_name)
        

    def retrieve(self, rep_query, limit=DEFAULT_LIMIT):
        results = self.idx.search(rep_query) #already sorted
        print('BMXIndex results len: ', len(results.keys))
        keys, scores = results.keys[:limit], results.scores[:limit]
        docnodes = [ScoreNode(doc_path=key, score=score, is_ref=True) for key, score in zip(keys, scores)]
        return docnodes
    
    
    