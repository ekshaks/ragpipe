'''
https://colab.research.google.com/gist/seldo/f6b3515db1f4dd7976d70d54054f6996/demo_insurance.ipynb#scrollTo=syTF15KRMhD8

Highlights:
- pdf to mmd with nougat
- fix table labels
- ragpipe config format - switch limit, prompts
- (optional) switch from dense to colbert. cutoff difference?

1. fix table section labels (niva.mmd -> niva-short.mmd)

relate annexure heading to table
precise spec - # some tables are labeled properly. use ### label \n\n table

The following is a snippet from a mmd document obtained by converting a PDF document automatically. Some tables are not linked to their labels properly. Rewrite this snippet so that all tables are properly placed in sections corresponding to their labels. Only show the diff between old and new markdown content. Output raw markdown.

2. build doc -> (section-content, meta: header_path) #partially hierarchy aware. 
#to get section.subsection -> recover from header_paths

3. build reps for section.* 
#TODO: colbert emb instead of dense -> clear cutoff

4. bridge: query <-> section-content. on match, output: header_path + section-content
'''

from pathlib import Path
from ragpipe.common import DotDict, printd
import yaml


def respond(query, docs_retrieved, prompt):
    docs_texts = '\n'.join([doc.get_text_content() for doc in docs_retrieved])
    prompt = prompt.format(documents=docs_texts, query=query)
    from ragpipe.llm_bridge import llm
    resp = llm.call_api(prompt, model='mistral')
    print(resp)

def build_data_model(md_file):
    from llama_index.core.node_parser import MarkdownNodeParser
    from llama_index.readers.file import FlatReader

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

def main():
    with open('examples/insurance.yml', 'r') as file:
        config = DotDict(yaml.load(file, Loader=yaml.FullLoader))
    
    D = build_data_model('examples/data/insurance/niva-short.mmd')
    printd(3, 'over build data model')

    queries = [
        "I just had a baby, is baby food covered?",
       "How is gauze used in my operation covered?"
    ]
    query_text = queries[1]

    from ragpipe.bridge import bridge_query_doc

    docs_retrieved = bridge_query_doc(query_text, D, config)
    #print(docs_retrieved) #response generator
    for doc in docs_retrieved: doc.show_li_node()
    respond(query_text, docs_retrieved, config['prompts']['qa2']) 


if __name__ == '__main__':
    main()