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

def main():
    from ragpipe.config import load_config
    config = load_config('examples/billionaires.yml', show=True)
    
    data_folder = config.etc['data_folder']
    assert Path(data_folder).exists(), f'Data folder not found. Please clone github.com/ragpipe/data and point config variable etc/data_folder in startups.yml to the data folder.'
    
    md_path = f'{data_folder}/billionaires.md'
    assert Path(md_path).exists(), f'Markdownfile not found: {md_path}!'

    D = build_data_model(md_path)
    printd(3, 'over build data model')

    query_text = config.queries[0]
    printd(1, f'Query: {query_text}')
    from ragpipe.bridge import bridge_query_doc

    docs_retrieved = bridge_query_doc(query_text, D, config)
    print('Documents: ', docs_retrieved) #response generator

    from ragpipe.llms import respond_to_contextual_query

    resp = respond_to_contextual_query(query_text, docs_retrieved, config.prompts['qa']) 
    print('Answer:\n', resp)



if __name__ == '__main__':
    main()
