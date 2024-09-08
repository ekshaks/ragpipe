try:
    from tqdm import tqdm
    import torch
    from torch.utils.data import DataLoader

    from colpali_engine.models.paligemma_colbert_architecture import ColPali
    from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
    from colpali_engine.utils.colpali_processing_utils import process_images, process_queries
    from colpali_engine.utils.image_from_page_utils import load_from_dataset
except Exception as e:
    print('To use colpali: please `pip install colpali-engine`')
    raise e

from ragpipe.encoders import Encoder
from ragpipe.common import printd



class ColpaliEnc(Encoder):
    @staticmethod
    def load_model(device="cpu"):
        from transformers import AutoProcessor

        model_name = "vidore/colpali-v1.2"
        model = ColPali.from_pretrained("vidore/colpaligemma-3b-pt-448-base", torch_dtype=torch.bfloat16, device_map=device).eval()
        model.load_adapter(model_name)
        model = model.eval()
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor

    def encode_queries(self, queries, batch_size=2):
        from PIL import Image
        model, processor = self.get_model()

        printd(2, '>> Colpali: Encoding Queries')

        dataloader = DataLoader(
            queries,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: process_queries(processor, x, Image.new("RGB", (448, 448), (255, 255, 255))),
        )

        qs = []
        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
                embeddings_query = model(**batch_query)
            qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))
        return qs
    
    def encode(self, docs, is_query=False):
        batch_size = self.config.batch_size
        if is_query:
            return self.encode_queries(docs, batch_size=batch_size)
        model, processor = self.get_model()

        images = docs
        page_embeddings = []
        dataloader = DataLoader(
            images,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: process_images(processor, x),
        )
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
                embeddings_doc = model(**batch_doc)
                page_embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    

        return page_embeddings
    
    
    def get_similarity_fn(self):

        def sim(doc_embeddings=None, query_embedding=None):
            scores = CustomEvaluator(is_multi_vector=True).evaluate([query_embedding], doc_embeddings)
            return scores[0]
         
        return sim

    @classmethod
    def from_config(cls, config):
        model_loader = \
        lambda: ColpaliEnc.load_model(device=config.device)

        return ColpaliEnc(name=config.name, mo_loader=model_loader, config=config, rep_type='multi_vector')
    