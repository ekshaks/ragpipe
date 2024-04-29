from pathlib import Path
from ragpipe.common import DotDict, printd
import yaml


def respond(query, docs_retrieved, prompt):
    docs_texts = '\n'.join([doc.get_text_content() for doc in docs_retrieved])
    prompt = prompt.format(documents=docs_texts, query=query)
    from ragpipe.llm_bridge import llm
    resp = llm.call_api(prompt, model='mistral')
    print(resp)

{"name":"SaferCodes","images":"https:\/\/safer.codes\/img\/brand\/logo-icon.png","alt":"SaferCodes Logo QR codes generator system forms for COVID-19","description":"QR codes systems for COVID-19.\nSimple tools for bars, restaurants, offices, and other small proximity businesses.","link":"https:\/\/safer.codes","city":"Chicago"}
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
    D = DotDict(documents=documents, doc_leaf_type='raw')
    return D

def main():
    with open('examples/startups.yml', 'r') as file:
        config = DotDict(yaml.load(file, Loader=yaml.FullLoader))
    
    D = build_data_model('examples/data/startups/startups_demo-small.json')
    #D = build_data_model('examples/data/startups/startups_demo.json')
    printd(1, '-==== over build data model')

    queries = [
       "healthcare",
       "fashion",
       "financial"
    ]
    query_text = queries[0]

    from ragpipe.bridge import bridge_query_doc

    docs_retrieved = bridge_query_doc(query_text, D, config)
    #print(docs_retrieved) #response generator
    printd(1, f'query: {query_text}')
    for doc in docs_retrieved: doc.show()
    #respond(query_text, docs_retrieved, config['prompts']['qa2']) 


if __name__ == '__main__':
    main()