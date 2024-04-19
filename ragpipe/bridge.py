from typing import List
from pathlib import Path
import os

from ragpipe.common import get_fpath_items

from .common import printd, load_func
from .common import DotDict, DEFAULT_LIMIT

def compute_representations(D, config):
    '''
    compute:
    recursively Q, transcript
    index:
    each field, rep name -> 
        - index yes? index_reps -> create indexed collection, store in vecdb
            - collection: list(vec), list(metadata) or df[vec, field*]
            - options: which field to include?
            - rep [fpath, rep_name] -> (fpath, collection)
        - index no? mem_reps[field, rep_name] -> (path, rep)

    '''
    from .rag_components import compute_rep

    def hash_field_repname(fpath, repname):
        return f'{fpath}#{repname}'
    
    def _compute_index_reps(fpath, D, C, is_query=False):
        #fpath: path to to-be-rep node in O, O: parent object, C: config
        _reps = {}
        for repname, properties in C.items():
            #printd(3, f'rep for : {repname}')
            props = DotDict(properties)
            rep_path_pairs = compute_rep(fpath, D, props=props, repname=repname, is_query=is_query)
            rep_key = hash_field_repname(fpath, repname)
            _reps[rep_key] = rep_path_pairs
        return _reps
        
    reps = {} #'Chunk.content' -> dense -> rep=(doc, <vec>)
    #field, rep_name -> (collection_path | rep)

    for field, repC in config.representations.items(): 
        printd(3, f'compute_index: field={field}, config={repC}')
        is_query = 'query' in field
        #fpath = field.split('.')[-1] if is_query else field
        fpath = field
        _reps = _compute_index_reps(fpath, D, repC, is_query=is_query)
        reps.update(_reps)


    return reps



def show_docs(docs):
    try:
        for doc in docs:
            print(doc.node.id_, doc.score, doc.node.metadata['file_path'], doc.node.text[:100])
    except:
        print(docs)


def compute_bridge_scores(reps, D, bridge_config):
    '''
    compute score expr for each pair of field-repname, limit by N
    assume bridge of form (mem rep - vecdb rep), use vecdb apis
    score['b1'] = [(doc, score)] 
    b1: 
     repnodes: Query.text#dense, chunk#dense
     limit: 10
    '''
    def get_reps(qkey, reps):
        keys = list(reps.keys())
        for key in keys:
            if qkey in key:
                #printd(3, f'get_reps found: {qkey}, {key}')
                return reps[key]

        raise ValueError(f'unable to find {qkey} in reps')


    from .docnode import ScoreNode

    docs_retrieved = {}
    printd(3, f'bridge config: {bridge_config}')
    printd(3, f'reps keys: {list(reps.keys())}')

    for bridge_name, properties in bridge_config.items():
        props = DotDict(properties)
        repkey1, repkey2 = map(lambda x: x.strip(), props.repnodes.split(','))
        printd(2, f'==== now bridging {repkey1}, {repkey2}')
        matchfn_key = props.get('matchfn')
        limit = props.get('limit') 
        limit = DEFAULT_LIMIT if limit is None else limit

        docs: List[ScoreNode]

        if matchfn_key is not None:
            rep1 = get_reps(repkey1, reps)
            rep2 = get_reps(repkey2, reps)
            matchfn = load_func(matchfn_key)
            docs: List[ScoreNode] = matchfn(rep1, rep2)
        else:
            from .rag_components import retriever_router
            #https://docs.llamaindex.ai/en/stable/examples/query_engine/CustomRetrievers.html
            query_index = get_reps(repkey1, reps) 
            doc_index = get_reps(repkey2, reps)
            docs: List[ScoreNode] = retriever_router(doc_index, D.query.text, query_index, limit=limit)
        
        for d in docs: d.load_docs(D)

        docs_retrieved[bridge_name] = docs

        #show_docs(docs)

    return docs_retrieved #may or not have scores associated with docs. normalize how?



def rank_paths(scores, rank_config):
    '''
    for each query rep_name: 
        - start from query, end at doc leaves? get score. rank docs by score
    ranked_scores[rep_name] = [(doc, score)]
    ranks[rep_name] = [doc-sc, doc-sc, doc-sc]

    results: List[Result] = fuse_ranks(ranked_scores, ranks) 
    #Result = [doc, score]*

    rank: 
        expr: b1
        limit: 10

    '''
    def eval_score_expr(expr, scores):
        #TODO: generalize! 
        #use expr to gen new scores for each doc common across all bridge_names. sort.
        return scores[expr] 
    rank_config = DotDict(rank_config)
    doc_with_scores = eval_score_expr(rank_config.expr, scores)
    return doc_with_scores


def bridge_query_doc(query_text, D, config):
    Q = DotDict(text=query_text)
    D.query = Q
    reps = compute_representations(D, config)
    path_scores = compute_bridge_scores(reps, D, config.bridge) #repNode1, repNode2 -> score.
    ranked_doc_with_scores = rank_paths(path_scores, config.rank) 
    return ranked_doc_with_scores
