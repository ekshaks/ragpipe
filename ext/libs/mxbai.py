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
            out_embeddings = np.unpackbits(out_embeddings, axis=-1).astype("int") #(B, 512)
        else:
            out_embeddings = embeddings
        #printd(3, f'mxlarge encode (ubinary): {out_embeddings.shape}')
        #printd(3, f'mxlarge encode (ubinary) unpacked: {out_embeddings.shape}')
        #print(out_embeddings)
        return out_embeddings*1.0 #turn Long to Float
    
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
            res = cos_sim(query_embedding, doc_embeddings).squeeze()
            #printd(4, 'mxbai: get sim: ', res.shape, res)
            return res
        return sim

    @classmethod
    def from_config(cls, config):
        model_loader = \
        lambda: SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=config.shape.size)

        return MXLarge(name=config.name, mo_loader=model_loader, config=config, rep_type='single_vector')
    