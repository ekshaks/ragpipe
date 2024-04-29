

from .docnode import ScoreNode
from typing import List
from collections import defaultdict

def reciprocal_rank_fusion(bridge2results, k=60):
    '''
    merges the results obtained from different bridges
    bridge2results = {bridge_k: List[ScoreNode]}
    '''
    doc_path_score = {}
    bridge2rank = defaultdict(dict)
    for bridge, results in bridge2results.items():
        results: List[ScoreNode]
        for rank, doc_node in enumerate(results):
            doc_path = doc_node.doc_path
            if doc_path not in doc_path_score:
                doc_path_score[doc_path] = 0
            doc_path_score[doc_path] += 1 / (rank + k)
            bridge2rank[doc_path][bridge] = rank + 1

    
    fused_results = [ScoreNode(doc_path=doc_path, score=score, bridge2rank=bridge2rank[doc_path])
                      for doc_path, score in doc_path_score.items()]
    fused_results = sorted(fused_results, key=lambda x: x.score, reverse=True)
    return fused_results
            