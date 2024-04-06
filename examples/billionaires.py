from pathlib import Path
from ragpipe.common import DotDict, printd



import yaml

def build_data_model(md_file):
    # read md as nodes / docs. encapsulate in desired object structure
    '''
    sections: List[section]
    section: {title: title, paras: List[para], subsections: List[subsection]}
    subsection: {title: title, paras: List[para]}
    para: text | table


    doc: List[section]
    section: {headerpath: path, text: text}
    '''

    from llama_index.core.node_parser import MarkdownNodeParser

    from llama_index.readers.file import FlatReader

    md_docs = FlatReader().load_data(Path(md_file))              
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(md_docs)

    sections = []
    for node in nodes:
        print(node.metadata)
        headers = [ v.strip() for key, v in node.metadata.items() if 'Header' in key]
        headerpath = ' > '.join(headers)
        #print(headerpath)
        sections.append(DotDict(headerpath=headerpath, node=node))
    D = DotDict(sections=sections)
    return D

def match_year(query_index: 'ObjectIndex', doc_index):
    from ragpipe.docnode import ScoreNode
    #print(query_index, doc_index)
    year = query_index.get_query_rep()['metadata']['year']
    docpath_scores = [ScoreNode(doc_path=doc_path, score=1.0) for doc_rep, doc_path in doc_index.items()
        if year in doc_rep]
    return docpath_scores

def respond(query, docs_retrieved, prompt):
    docs_texts = '\n'.join([doc.get_text_content() for doc in docs_retrieved])
    prompt = prompt.format(documents=docs_texts, query=query)
    from ragpipe.llm_bridge import llm
    resp = llm.call_api(prompt, model='mistral')
    print(resp)



def main():
    with open('examples/billionaires.yml', 'r') as file:
        config = DotDict(yaml.load(file, Loader=yaml.FullLoader))
    
    D = build_data_model('examples/data/billionaires.md')
    printd(3, 'over build data model')

    queries = [
        "What's the net worth of the second richest billionaire in 2023?",
        "How many billionaires were there in 2021?"
    ]
    query_text = queries[0]

    from ragpipe.bridge import bridge_query_doc

    docs_retrieved = bridge_query_doc(query_text, D, config)
    #print(docs_retrieved) #response generator
    respond(query_text, docs_retrieved, config['prompts']['qa']) 




if __name__ == '__main__':
    main()
