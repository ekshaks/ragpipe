from pathlib import Path
from ragpipe.common import DotDict, printd
from ragpipe.prompts import eval_template

class Workflow:
    #{"name":"SaferCodes","images":"https:\/\/safer.codes\/img\/brand\/logo-icon.png","alt":"SaferCodes Logo QR codes generator system forms for COVID-19","description":"QR codes systems for COVID-19.\nSimple tools for bars, restaurants, offices, and other small proximity businesses.","link":"https:\/\/safer.codes","city":"Chicago"}
    # 
    def init(self, config_fname, query_id=0):
        from ragpipe.config import load_config
        config = load_config(config_fname, show=False)
        data_folder = config.etc['data_folder']
        
        assert Path(data_folder).exists(), f'Data folder not found. Please clone github.com/ragpipe/data and point config variable etc/data_folder in startups.yml to the data folder.'
        #json_path = f'{data_folder}/startups/startups_demo-vsmall.json'
        json_path = f'{data_folder}/startups/startups_demo.json'
        assert Path(json_path).exists(), f'Data JSON file not found: {json_path}!'

        query_text = config.queries[query_id]
        return config, json_path, query_text
    
    def build_data_model(self, jsonl_file, config):

        import jsonlines
        documents = []
        with jsonlines.open(jsonl_file) as reader:
            for obj in reader: #name, images, alt, description, link
                description = obj['description'].strip()
                if description == '':
                    obj['description'] =obj['alt']
                documents.append(obj)
        
        max_docs = config.etc.get('max_docs') or None
        documents = documents[:max_docs] # restrict to first `max_docs` for quick testing

        print(f'Num Documents = {len(documents)}')
        D = DotDict(documents=documents)
        return D

    def respond(self, query, docs_retrieved, prompt_templ, llm_model):
        docs_texts = '\n'.join([doc.get_text_content() for doc in docs_retrieved])
        prompt = eval_template(prompt_templ, documents=docs_texts, query=query)
        from ragpipe.llms import cloud_llm
        resp = cloud_llm(prompt, model=llm_model)
        return resp


    def run(self, respond_flag=False):
        parent = Path(__file__).parent; 
        config_fname = f'{parent}/startups.yml'
        config, json_path, query_text = self.init(config_fname)
       
        D = self.build_data_model(json_path, config)
        printd(1, '-==== over build data model')

        from ragpipe import Retriever
        docs_retrieved = Retriever(config).eval(query_text, D)
        #print(docs_retrieved) #response generator
        printd(1, f'query: {query_text}')
        for doc in docs_retrieved: doc.show()

        if respond_flag:
            return self.respond(query_text, docs_retrieved, config.prompts['qa2'], config.llm_models.default) 
        else:
            return docs_retrieved

if __name__ == '__main__':
    Workflow().run()