from pathlib import Path
from ragpipe.common import DotDict, printd

# This is a quickstart ragpipe project to get you started. 
# Edit the functions below to customize.

class Workflow:
    # 
    def init(self, query_id=0):
        from ragpipe.config import load_config
        parent = Path(__file__).parent
        config = load_config(f'{parent}/project.yml', show=True)
        data_folder = config.etc['data_folder']
        
        assert Path(data_folder).exists(), f'Data folder not found. Please clone github.com/ragpipe/data and point config variable etc/data_folder in startups.yml to the data folder.'
        json_path = f'{data_folder}/project/data.json'
        assert Path(json_path).exists(), f'Data JSON file not found: {json_path}!'

        query_text = config.queries[query_id]
        return config, json_path, query_text
    
    def build_data_model(self, jsonl_file):

        import jsonlines
        documents = []
        with jsonlines.open(jsonl_file) as reader:
            for obj in reader: 
                # update obj
                documents.append(obj)
        D = DotDict(documents=documents)
        return D

    def respond(self, query, docs_retrieved, prompt_templ, llm_model):
        from ragpipe.llms import respond_to_contextual_query
        resp = respond_to_contextual_query(query, docs_retrieved, prompt_templ, model=llm_model)
        return resp
    
        # Alternatively, create the prompt manually and call an LLM
        # from ragpipe.prompts import eval_template
        # docs_texts = '\n'.join([doc.get_text_content() for doc in docs_retrieved])
        # prompt = eval_template(prompt_templ, documents=docs_texts, query=query)
        # from ragpipe.llms import llm_router
        # resp = llm_router(prompt, model=llm_model)


    def run(self, respond_flag=False):
        config, json_path, query_text = self.init()
       
        D = self.build_data_model(json_path)
        printd(1, '-==== over build data model')

        from ragpipe import Retriever
        docs_retrieved = Retriever(config).eval(query_text, D)

        printd(1, f'query: {query_text}')
        for doc in docs_retrieved: doc.show()

        if respond_flag:
            return self.respond(query_text, docs_retrieved, config.prompts['qa'], config.llm_models.__default__) 
        else:
            return docs_retrieved

if __name__ == '__main__':
    Workflow().run()