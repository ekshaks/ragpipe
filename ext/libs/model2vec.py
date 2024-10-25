from ragpipe.encoders import Encoder
from ragpipe.common import printd

try:
    from model2vec import StaticModel
    from sentence_transformers.util import cos_sim

except Exception as e:
    print('To use model2vec: please `pip install model2vec sentence_transformers`')
    raise e


class Model2Vec(Encoder):
    def load_model(device='cpu'):
        model_name = "minishlab/M2V_base_output"
        model = StaticModel.from_pretrained(model_name)
        return model
    
    def encode(self, docs, is_query=False) : 
        model = self.get_model()
        #embeddings = model.encode(["It's dangerous to go alone!", "It's a secret to everybody."])
        return model.encode(docs)
        
    def get_similarity_fn(self):

        def sim(doc_embeddings, query_embedding):
            res = cos_sim(query_embedding, doc_embeddings).squeeze()
            #printd(4, 'model2vec: get sim: ', res.shape, res)
            return res
        return sim

    @classmethod
    def from_config(cls, config):
        model_loader = \
        lambda: Model2Vec.load_model(device=config.device)

        return Model2Vec(name=config.name, mo_loader=model_loader, config=config, rep_type='single_vector')