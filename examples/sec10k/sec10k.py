from pathlib import Path
from ragpipe.common import DotDict, printd
from ragpipe.prompts import eval_template
from ragpipe.llms import LLMOp

def concat_files(md_reps):
    """
    Concatenates the contents of multiple markdown files.

    Args:
        md_reps (list): A list of paths to markdown files.

    Returns:
        str: The concatenated content of the markdown files.
    """
    concatenated_content = ""

    # Loop through each markdown file
    for file_path in md_reps:
        try:
            # Attempt to open and read the file
            with open(file_path, 'r') as file:
                # Read the file content and concatenate it
                concatenated_content += file.read()
        except FileNotFoundError:
            # Handle the case when the file is not found
            print(f"File {file_path} not found. Skipping...")
        except Exception as e:
            # Handle any other exceptions
            print(f"An error occurred: {e}")

    return concatenated_content

def filter_llm(input_reps):
    #https://blog.dottxt.co/extracting-financial-data.html
    
    import outlines
    import numpy as np
    model = outlines.models.transformers("microsoft/Phi-3.5-mini-instruct")

    # Classification function
    yesno = outlines.generate.choice(model, ['Yes', 'Maybe', 'No'])

    # Requesting a classification from the model
    results = [ yesno(
        f"Is the following document about an income statement? Document: {rep}"
    ) for rep in input_reps]
    results = np.array(results)
    inputs = np.array(input_reps)

    filtered = inputs[results == 'Yes']
    return filtered


def map_agg_vlm(input_reps, llmops):
    from ragpipe.llms import cloud_vlm
    #print(f'processing {image_paths}')    
    image_paths = [rep.image_path for rep in input_reps]
    return cloud_vlm(image_paths, op=llmops.combined_op) #map, agg in one step

def map_agg_llm(md_paths, config):
    '''
    mdpaths = filter(tfilter_op, mdpaths)
    results = map(tmap_op, mdpaths)
    result = agg(tagg_op, results)
    '''


class Workflow:
    # 
    def init(self, query_id=0):
        from ragpipe.config import load_config
        parent = Path(__file__).parent
        config = load_config(f'{parent}/sec10k.yml', show=True)
        self.config = config

        data_folder = Path(config.etc['data_folder'])
        assert data_folder.exists(), f'Data folder not found. Please clone github.com/ragpipe/data and point config variable etc/data_folder in .yml to the data folder.'
        pdf_path = list((data_folder/'sec10k').glob('*.pdf'))[0]
        query_text = config.queries[query_id]
        return config, pdf_path, query_text
    
    def build_data_model(self, pdf_path, redo=False):
        from ragpipe.ingest.parsers.pdf_parsers import pdf_to_images
        output_dir = pdf_path.parent / 'images'
        output_dir.mkdir(exist_ok=True, parents=True)
        if redo:
            pdf_to_images(pdf_path, output_dir)
        images = [dict(image_path=fpath) for fpath in output_dir.glob('*.png')]
        D = DotDict(images=images)
        return D

    def respond(self, query, docs_retrieved, prompt_templ, llm_model):
        from ragpipe.llms import respond_to_contextual_query
        resp = respond_to_contextual_query(query, docs_retrieved, prompt_templ, model=llm_model) 
        #resp = respond_to_contextual_query(query, docs_retrieved, prompt_templ, config=self.config) # to pickup the default model from config.llm_models
        return resp
    
        # Alternatively, create the prompt manually and call an LLM
        # from ragpipe.prompts import eval_template
        # docs_texts = '\n'.join([doc.get_text_content() for doc in docs_retrieved])
        # prompt = eval_template(prompt_templ, documents=docs_texts, query=query)
        # from ragpipe.llms import llm_router
        # resp = llm_router(prompt, model=llm_model)


    def run(self, respond_flag=False, vlm=False):
        config, json_path, query_text = self.init()
       
        D = self.build_data_model(json_path)
        printd(1, '-==== over build data model')

        from ragpipe import Retriever
        docs_retrieved = Retriever(config).eval(query_text, D)

        printd(1, f'query: {query_text}')
        for doc in docs_retrieved: doc.show() #.images[].image_path

        image_reps = [DotDict(image_path=doc.get_file_path()) 
                for doc in docs_retrieved]
        
        image_prompt = eval_template(config.prompts['vqa1'], query=query_text)

        if vlm: #use mm llm to extract answer from images
            ops = DotDict(
                combined_op= LLMOp(prompt=image_prompt, model=config.llm_models['llmv2'], 
                                params=DotDict(max_images_per_call=1)
                )
            )
            res = map_agg_vlm(image_reps, ops)
            printd(1, res)
        else: 
            # images -> md, then use text llm
            from ragpipe.ingest.parsers.docling_parser import image2md
            from ragpipe.llms import llm_router

            md_reps = image2md(image_reps)
            #res = map_agg_llm(query_text, md_reps, ops)
            md_agg_rep = concat_files(md_reps)
            prompt = eval_template(config.prompts['qa1'], query=query_text, documents=md_agg_rep)
            res = llm_router(prompt, model=config.llm_models['__default__'])
            printd(1, res)

def test_vlm():
    from pathlib import PosixPath
    image_paths = [PosixPath('data/sec10k/images/page_50.png'), PosixPath('data/sec10k/images/page_78.png'), PosixPath('data/sec10k/images/page_39.png'), PosixPath('data/sec10k/images/page_53.png'), PosixPath('data/sec10k/images/page_36.png'), PosixPath('data/sec10k/images/page_54.png'), PosixPath('data/sec10k/images/page_55.png'), PosixPath('data/sec10k/images/page_58.png'), PosixPath('data/sec10k/images/page_38.png'), PosixPath('data/sec10k/images/page_44.png')]
    image_paths = [DotDict(image_path=p) for p in image_paths]
    config, _, query = Workflow().init()
    image_prompt = eval_template(config.prompts['vqa1'], query=query)

    ops = DotDict(
            combined_op= LLMOp(prompt=image_prompt, model=config.llm_models['llmv2'], 
                            params=DotDict(max_images_per_call=1)
            )
        )
    res = map_agg_vlm(image_paths, ops)
    print(res)

if __name__ == '__main__':
    Workflow().run(vlm=True)
    #test_vlm()