from typing import List

from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.clip import ClipEmbedding

from . import llm_bridge
from .common import get_fpath_items, printd, DEFAULT_LIMIT, DotDict
from .colbert import Colbert
from .docnode import ScoreNode


class RPIndex(): #rag pipe index
    #TODO: add storage backend
    def __init__(self, encoder, doc_embeddings, doc_paths=[], is_query=False):
        self.encoder = encoder
        self.doc_embeddings = doc_embeddings
        self.doc_paths = doc_paths
        self.is_query = is_query
    
    def get_query_rep(self):
        assert self.is_query, 'cant get rep from non-query index'
        return self.doc_embeddings[0]

    def retrieve(self, rep_query, limit=DEFAULT_LIMIT):
        scores = self.encoder.compute_similarity_embeddings(rep_query, self.doc_embeddings)
        #scores: len(self.doc_embeddings)
        printd(2, f'retrieve: {scores}')
        doc_nodes = [ScoreNode(doc_path=doc_path, is_ref=True, score=score) 
         for doc_path, score in zip(self.doc_paths, scores)]
        doc_nodes = sorted(doc_nodes, key=lambda x: x.score, reverse=True)[:limit]
        return doc_nodes

class VectorStoreIndexPath(VectorStoreIndex):
    def __init__(self, *args, **kwargs):
        self.doc_paths = kwargs.pop('doc_paths')
        super().__init__(*args, **kwargs)

class ObjectIndex():
    def __init__(self, reps, paths, is_query=False):
        self.reps, self.paths = reps, paths
        self.is_query = is_query
    def get_query_rep(self):
        assert self.is_query, 'Index does not store query rep'
        return self.reps[0]
    def items(self):
        '''Returns a list of tuples (rep, path) contained in the index
        '''
        return zip(self.reps, self.paths)
    
    def __str__(self):
        return(f'\nObjectIndex: \n reps = {self.reps} \n paths = {self.paths}\n')
    
def embed_index(encoder, items, item_paths, is_query=False):
    if is_query: #based on data type, call different encoders
        reps = embed_index_generic(encoder, items, item_paths, is_query=is_query)
    else:
        #TODO: lookup a cached collection -- by path?
        assert isinstance(items, list) and len(items) > 0, f"items is {type(items)}"
        item_type = type(items[0]).__name__
        #printd(1, 'embed_index : ' +item_type)
        if 'TextNode' in item_type:
            #reps = VectorStoreIndex(items, embed_model=encoder, show_progress=True)
            reps = VectorStoreIndexPath(items, doc_paths=item_paths, embed_model=encoder, show_progress=True)
        else:
            reps = VectorStoreIndexPath.from_documents(items, doc_paths=item_paths, embed_model=encoder, show_progress=True)
    return reps

def embed_index_generic(encoder, items, item_paths, is_query=False):
    #node = TextNode(text=items[0], id_="_query_")

    item_type = type(items[0]).__name__
    if item_type != 'str': #handle LI text nodes. TODO: what if LI documents?
        assert 'TextNode' in item_type, f'Cannot handle item type {item_type}'
        items = [item.text for item in items]

    reps = RPIndex(encoder=encoder, 
                   doc_embeddings=[encoder.get_text_embedding(item) for item in items],
                   doc_paths=item_paths,
                   is_query=is_query
                   )
    return reps


def compute_rep(fpath, D, encoder_name, repname=None, is_query=False) -> 'List[rep]':
    #TODO: if fpath, encoder is in cache, return cache RPIndex or VectorStoreIndex
    
    items_path_pairs = get_fpath_items(fpath, D)
    items, item_paths = items_path_pairs.els, items_path_pairs.paths

    if encoder_name.startswith('llm'):
        reps = llm_bridge.transform(items, encoder_name, is_query=is_query)
    elif encoder_name == 'passthrough':
        print('computing rep passthrough')
        _repname = repname[1:] # _header -> header
        reps = [item[_repname] for item in items]
    else:
        match encoder_name:
            case "bm25":
                from .encoders import BM25
                reps = items if is_query else BM25(items) #no change is query, else make corpus
                reps = RPIndex(encoder='bm25', doc_embeddings=reps, is_query=is_query)
                
            case "BAAI/bge-small-en-v1.5":
                text_embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
                reps = embed_index(text_embed_model, items, item_paths, is_query=is_query)
            
            case "colbert-ir/colbertv2.0":
                text_embed_model = Colbert(model=encoder_name, tokenizer=encoder_name)
                reps = embed_index_generic(text_embed_model, items, item_paths, is_query=is_query)
            
            case "jinaai/jina-colbert-v1-en":
                text_embed_model = Colbert(model=encoder_name, tokenizer=encoder_name, max_length=8192)
                reps = embed_index_generic(text_embed_model, items, item_paths, is_query=is_query)
            
            case "ViT-B/32":
                image_embed_model = ClipEmbedding(model_name="ViT-B/32")
                reps = embed_index(image_embed_model, items, item_paths, is_query=is_query)
            
            case _:
                raise ValueError(f"unknown encoder: {encoder_name}")
    #print(type(items[0]), items[0])

    match reps:
        #case list(): #one rep per item
        #    rep_path_pairs = list(zip(item_reps, item_paths))
        case RPIndex(): #single collection. RPIndex (has paths already)
            rep_path_pairs = reps
        case VectorStoreIndexPath(): #single collection. enumerate paths or keep a single jq path
            rep_path_pairs = reps #[ (reps, path) for path in item_paths ]
        case list():
            rep_path_pairs = ObjectIndex(reps=reps, paths=item_paths, is_query=is_query) #
        case _:
            raise ValueError(f"Unknown rep: {reps}")
    return rep_path_pairs

def retriever_router(doc_index, query_text, query_index, limit=10): #TODO: rep_query_path -> query_index
    '''
    Depending on the type of index, retrieve the doc nodes relevant to a query
    '''
    from llama_index.core import QueryBundle
    from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
    #from llama_index.core.schema import NodeWithScore
           #rp = (rep_value, rep_item_path)
    print('doc_index type', type(doc_index), type(query_index))
    #print(doc_index)
    #print(query_index)
    assert(isinstance(query_index, RPIndex)), f'query_index type= {type(query_index)} '
    rep_query = query_index.get_query_rep()

    #TODO: remove below
    match doc_index:
        case RPIndex():
            index = doc_index
        case VectorStoreIndexPath():
            index = doc_index
        case _:
            raise ValueError(f'unknown index type {doc_index}')
    #    index = doc_index[0] 
    #    rep_query = query_index[0][0] # pick first of reps, then pick rep value
    
    match index:
        case VectorStoreIndexPath():
            query_bundle = QueryBundle(query_str=query_text, embedding=rep_query)
            retriever = VectorIndexRetriever(index=index, similarity_top_k=limit)
            li_nodes = retriever.retrieve(query_bundle)
            #vector_ids = {n.node.node_id for n in vector_nodes}
            doc_nodes = [ScoreNode(li_node=n, score=n.score) for n in li_nodes]

            return doc_nodes

        case RPIndex():
            doc_nodes: List[ScoreNode] = index.retrieve(rep_query, limit=limit) #only refs
            return doc_nodes
        
        case _:
            raise NotImplementedError(f'unknown index: {index}')
