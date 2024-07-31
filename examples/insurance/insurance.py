# adapted from https://colab.research.google.com/gist/seldo/f6b3515db1f4dd7976d70d54054f6996/demo_insurance.ipynb#scrollTo=syTF15KRMhD8

from pathlib import Path
from ragpipe.common import DotDict, printd
from ragpipe.config import load_config


def respond(query, docs_retrieved, prompt):
    docs_texts = '\n'.join([doc.get_text_content() for doc in docs_retrieved])
    prompt = prompt.format(documents=docs_texts, query=query)
    #from ragpipe.llm_bridge import local_llm
    #resp = local_llm.call_api(prompt, model='mistral')
    from ragpipe.llms import groq_llm
    resp = groq_llm(prompt)
    print(resp)

def build_data_model(md_file):
    from llama_index.core.node_parser import MarkdownNodeParser
    from llama_index.readers.file import FlatReader

    print('building data model')
    md_docs = FlatReader().load_data(Path(md_file))              
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(md_docs)

    sections = []
    for node in nodes:
        print("*************", node.metadata)
        #print(node.text)
        headers = [ v.strip() for key, v in node.metadata.items() if 'Header' in key]
        headerpath = ' > '.join(headers)
        #print(headerpath)
        sections.append(DotDict(headerpath=headerpath, node=node))
    D = DotDict(sections=sections)
    return D

def main(respond_flag=False):
    #with open('examples/insurance.yml', 'r') as file:
    #    config = DotDict(yaml.load(file, Loader=yaml.FullLoader))
    parent = Path(__file__).parent
    config = load_config(f'{parent}/insurance.yml', show=True)
    data_folder = config.etc['data_folder']
    md_path = Path(f'{data_folder}/insurance/niva-short.mmd')
    assert Path(md_path).exists(), f'Data folder not found. Please clone github.com/ragpipe/data and point config variable etc/data_folder to the data folder.'

    D = build_data_model(md_path)
    printd(3, 'over build data model')

    query_text = config.queries[1]

    from ragpipe.bridge import bridge_query_doc

    docs_retrieved = bridge_query_doc(query_text, D, config)

    for doc in docs_retrieved: doc.show()

    if respond_flag:
        return respond(query_text, docs_retrieved, config.prompts['qa2']) 
    else:
        return docs_retrieved


if __name__ == '__main__':
    main(respond_flag=True)