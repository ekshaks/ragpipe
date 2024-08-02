from pathlib import Path
from ragpipe.common import DotDict, printd
from ragpipe.prompts import eval_template


def respond(query, docs_retrieved, prompt_templ, llm_model):
    docs_texts = '\n'.join([doc.get_text_content() for doc in docs_retrieved])
    prompt = eval_template(prompt_templ, documents=docs_texts, query=query)
    from ragpipe.llms import cloud_llm
    resp = cloud_llm(prompt, model=llm_model)
    return resp

#{"name":"SaferCodes","images":"https:\/\/safer.codes\/img\/brand\/logo-icon.png","alt":"SaferCodes Logo QR codes generator system forms for COVID-19","description":"QR codes systems for COVID-19.\nSimple tools for bars, restaurants, offices, and other small proximity businesses.","link":"https:\/\/safer.codes","city":"Chicago"}
# 
def build_data_model(jsonl_file):

    import jsonlines
    documents = []
    with jsonlines.open(jsonl_file) as reader:
        for obj in reader: #name, images, alt, description, link
            description = obj['description'].strip()
            if description == '':
                obj['description'] =obj['alt']
            documents.append(obj)
    D = DotDict(documents=documents)
    return D

def main(respond_flag=False):

    from ragpipe.config import load_config
    parent = Path(__file__).parent
    config = load_config(f'{parent}/startups.yml', show=True)
    data_folder = config.etc['data_folder']
    
    assert Path(data_folder).exists(), f'Data folder not found. Please clone github.com/ragpipe/data and point config variable etc/data_folder in startups.yml to the data folder.'
    json_path = f'{data_folder}/startups/startups_demo-vsmall.json'
    assert Path(json_path).exists(), f'Data JSON file not found: {json_path}!'

    D = build_data_model(json_path)
    printd(1, '-==== over build data model')

    queries = config.queries

    query_text = queries[0]

    from ragpipe.bridge import bridge_query_doc

    docs_retrieved = bridge_query_doc(query_text, D, config)
    #print(docs_retrieved) #response generator
    printd(1, f'query: {query_text}')
    for doc in docs_retrieved: doc.show()

    if respond_flag:
        return respond(query_text, docs_retrieved, config.prompts.qa2, config.llm_models.default) 
    else:
        return docs_retrieved

if __name__ == '__main__':
    main(respond_flag=False)