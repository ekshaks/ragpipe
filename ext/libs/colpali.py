from pathlib import Path
from typing import List, cast

try:
    from tqdm import tqdm
    import torch
    from torch.utils.data import DataLoader
    from PIL import Image
    from colpali_engine.models import ColQwen2, ColQwen2Processor

    from colpali_engine.models import ColPali
    from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
    from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
    from colpali_engine.utils.torch_utils import ListDataset, get_torch_device

    # from colpali_engine.models.paligemma_colbert_architecture import ColPali
    # from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
    # from colpali_engine.utils.colpali_processing_utils import process_images, process_queries
    # from colpali_engine.utils.image_from_page_utils import load_from_dataset
except Exception as e:
    print('To use colpali: please `pip install -U colpali-engine`')
    raise e

from ragpipe.encoders import Encoder
from ragpipe.common import printd

from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)  # assuming PIL Image
        return image

    def __len__(self):
        return len(self.image_paths)

class ColpaliEnc(Encoder):
    @staticmethod
    def load_model(device="cpu", model_name="vidore/colpali-v1.2"):
        if 'colpali' in model_name:
            model = ColPali.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
            ).eval()

            processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))

        elif 'colqwen' in model_name:
            model = ColQwen2.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device
            ).eval()
            processor = ColQwen2Processor.from_pretrained(model_name)
        elif 'colsmol' in model_name:
            from colpali_engine.models import ColIdefics3, ColIdefics3Processor

            model = ColIdefics3.from_pretrained(
                    "vidore/colsmolvlm-alpha",
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    attn_implementation="eager" #"flash_attention_2" or eager
                ).eval()
            processor = ColIdefics3Processor.from_pretrained("vidore/colsmolvlm-alpha")

        return model, processor

    def batch_encode(self, dataloader, model):
        qs: List[torch.Tensor] = []
        for batch_query in tqdm(dataloader):
            with torch.no_grad():
                batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
                embeddings_batch = model(**batch_query)
            qs.extend(list(torch.unbind(embeddings_batch.to("cpu"))))
        return qs

    def encode(self, docs, is_query=False):
        if isinstance(docs[0], (Path, Image.Image, str)):
            pass
        else:
            raise ValueError("Unsupported data type. Should be string or a Path-like object or a PIL Image.")
        
        model, processor = self.get_model()
        batch_size = self.config.batch_size

        # Process the inputs

        if is_query:
            assert isinstance(docs[0], str)
            #batch_items = processor.process_queries(docs).to(model.device)
            dataloader = DataLoader(
                dataset=ListDataset[str](docs),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda x: processor.process_queries(x),
            )

        else:
            #assert isinstance(docs[0], Image.Image)
            #batch_items = processor.process_images(docs).to(model.device)
            dataset = ImageDataset(docs)

            dataloader = DataLoader(
                #dataset=ListDataset[str](docs),
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda x: processor.process_images(x),
            )
        # Forward pass
        # with torch.no_grad():
        #     embeddings = model(**batch_items)
        embeddings = self.batch_encode(dataloader, model)

        return embeddings


    
    def get_similarity_fn(self):
        processor = ColQwen2Processor.from_pretrained(self.name)

        def sim(doc_embeddings=None, query_embedding=None):
            scores = processor.score_multi_vector([query_embedding], doc_embeddings)
            #scores = CustomEvaluator(is_multi_vector=True).evaluate([query_embedding], doc_embeddings)
            return scores[0]
         
        return sim

    @classmethod
    def from_config(cls, config):
        model_loader = \
        lambda: ColpaliEnc.load_model(device=config.device, model_name=config.name)

        return ColpaliEnc(name=config.name, mo_loader=model_loader, config=config, rep_type='multi_vector')
    