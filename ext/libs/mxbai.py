from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings
from ragpipe.encoders import Encoder
from ragpipe.common import printd
import numpy as np


class MXLarge(Encoder):
    def encode(self, docs, is_query=False):
        model = self.get_model()
        embeddings = model.encode(docs, normalize_embeddings=True)
        #printd(3, f'mxlarge encode: {embeddings.shape}') (B, 512)
        
        if self.config.shape.dtype == 'ubinary':
            out_embeddings = quantize_embeddings(embeddings, precision=self.config.shape.dtype) #(B, 64)
        else:
            out_embeddings = embeddings
        #printd(3, f'mxlarge encode (ubinary): {out_embeddings.shape}')
        return out_embeddings
    
    def encode_parallel(self, docs, is_query=False):
        model = self.get_model()
        pool = model.start_multi_process_pool()
        emb = model.encode_multi_process(docs, pool, normalize_embeddings=True)
        print("Embeddings computed. Shape:", emb.shape)
        model.stop_multi_process_pool(pool)
    
    def get_similarity_fn(self):
        #printd(3, f'get_similarity_fn: {self.name}')
        from sentence_transformers.util import cos_sim

        def sim(doc_embeddings=None, query_embedding=None):
            
            if self.config.shape.dtype == 'ubinary':
                #unpack, then cos_sim. TODO: improve
                query_embedding = np.unpackbits(query_embedding, axis=-1).astype("int")*1.0 #(B, 512)
                doc_embeddings = [np.unpackbits(d, axis=-1).astype("int")*1.0 for d in doc_embeddings]

            res = cos_sim(query_embedding, doc_embeddings).squeeze()
            #printd(4, 'mxbai: get sim: ', res.shape, res)
            return res
        
        def sim2(doc_embeddings=None, query_embedding=None):
            assert self.config.shape.dtype == 'ubinary', 'sim2 only applied to binary-quantized vectors'
            
            #hamming distance on compressed rep
            xor_result = np.bitwise_xor(doc_embeddings, query_embedding)
            #print('mxbai sim2 xor result: ', xor_result.shape)
            hamming_distance = np.unpackbits(xor_result, axis=-1).astype(int).sum(axis=-1)
            #print('mxbai sim2: hamm dist ', hamming_distance.shape, hamming_distance)
            return -hamming_distance + np.max(hamming_distance)

        return sim

    @classmethod
    def from_config(cls, config):
        model_loader = \
        lambda: SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=config.shape.size)

        return MXLarge(name=config.name, mo_loader=model_loader, config=config, rep_type='single_vector')
    