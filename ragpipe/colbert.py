import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
from .common import printd

#TODO: onnx model, 
class Colbert:
    def __init__(self, model:str ="colbert-ir/colbertv2.0", 
                tokenizer:str = "colbert-ir/colbertv2.0",
                max_length:int = 512
    ):
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self._model = AutoModel.from_pretrained(model)
        self.MAX_LENGTH = max_length
        print('Loaded Colbert model')

    def get_text_embedding(self, text: str):
        printd(1, 'embedding with Colbert')
        tokens = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=self.MAX_LENGTH)
        text_embedding = self._model(**tokens).last_hidden_state #1, docl, d=768
        return text_embedding
    
    def compute_relevance_scores(self, query_embeddings: '1,ql,d', document_embeddings: 'b,dl,d'):
        """
        Compute relevance scores for top-k documents given a query.
        
        :param query_embeddings: Tensor representing the query embeddings, shape: [num_query_terms, embedding_dim] (or 1,ql,d -- added ragpipe)
        :param document_embeddings: Tensor representing embeddings for k documents, shape: [k, max_doc_length, embedding_dim]
        :param k: Number of top documents to re-rank
        :return: Sorted document indices based on their relevance scores

        TODO: couldn't get this jina emb to work! instead of matmul, cosine similarity seems to work better
        """
        
        # Ensure document_embeddings is a 3D tensor: [k, max_doc_length, embedding_dim]
        # Pad the k documents to their maximum length for batch operations
        # Note: Assuming document_embeddings is already padded and moved to GPU
        
        # Compute batch dot-product of Eq (query embeddings) and D (document embeddings)
        # Resulting shape: [k, num_query_terms, max_doc_length]

        if len(query_embeddings.size()) == 2: #for ragpipe
            query_embeddings = query_embeddings.unsqueeze(0) #1,ql,d

        scores = torch.matmul(query_embeddings, document_embeddings.transpose(1, 2))
        
        # Apply max-pooling across document terms (dim=2) to find the max similarity per query term
        # Shape after max-pool: [k, num_query_terms]
        max_scores_per_query_term = scores.max(dim=2).values
        
        # Sum the scores across query terms to get the total score for each document
        # Shape after sum: [k]
        total_scores: 'b' = max_scores_per_query_term.sum(dim=1)
        printd(2, f'compute relevance colbert {total_scores}')
        return total_scores.tolist()
        # Sort the documents based on their total scores
        #sorted_indices = total_scores.argsort(descending=True)
        
        #return sorted_indices
    
    def compute_similarity_embedding(self, query_embedding=None, document_embedding=None):
            printd(3, f'compute_sim_emb: {query_embedding.shape}, {document_embedding.shape}')
            assert query_embedding is not None and document_embedding is not None

            # Query: b,ql,d -> b,ql,1,d
            # D: b,docl,d -> b,1,docl,d

            sim_matrix = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(2), document_embedding.unsqueeze(1), dim=-1
            )
            sim_matrix: 'b,ql,docl'

            max_sim_scores, _ = torch.max(sim_matrix, dim=2) #b, ql (max over doc length)
            #print(max_sim_scores)
            score = torch.mean(max_sim_scores, dim=1) #b (mean over query length)
            return score.item()

    def compute_similarity_embeddings(self, query_embedding=None, doc_embeddings=None):
        # Query: 1,ql,d
        # D: [1,docl,d]*
        if self.MAX_LENGTH == 512:
            scores = [ self.compute_similarity_embedding(query_embedding=query_embedding, document_embedding=demb)
                       for demb in doc_embeddings]
        else:
            assert False, 'compute_relevance scores needs debugging'
            #doc_embeddings: 'b,docl,d' = torch.stack(doc_embeddings)
            #scores = self.compute_relevance_scores(query_embedding, doc_embeddings)
            import itertools
            scores = [ self.compute_relevance_scores(query_embedding, demb) for demb in doc_embeddings]
            scores = list(itertools.chain.from_iterable(scores))

        return scores
    
    def compute_similarity_text(self, query: str, documents_text_list: List[str]) -> List[float]:
        query_embedding = self.get_text_embedding(query)

        rerank_score_list = []

        #TODO: pad doc tokens and batch process
        for document_text in documents_text_list:
            document_embedding = self.get_text_embedding(document_text)
            score = self.compute_similarity_embedding(query_embedding, document_embedding)
            rerank_score_list.append(score)

        return rerank_score_list


def test_colbert():
    colbert = Colbert()
    query = 'the game of cricket'
    docs = [
        'the next cricket match is tomorrow',
        'we should play tennis tomorrow',
        'elections are due soon'
    ]
    scores = colbert.compute_similarity(query, docs)
    print(scores)
    

if __name__ == '__main__':
    test_colbert()

