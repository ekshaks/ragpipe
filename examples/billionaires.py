from pathlib import Path
from ragpipe.common import DotDict, printd

def match_year(query_index, doc_index):
    from ragpipe.docnode import ScoreNode
    from ragpipe.common import tfm_docpath

    #print('Query Index: ', query_index)
    #print('Doc Index: ', doc_index)
    year: int = query_index.get_query_rep()['metadata']['year']
    print(f'year = {year}')
    # rule based bridge. for the doc paths filtered by the bridge, pick ..text as the new docpath
    docpath_scores = [ScoreNode(doc_path=tfm_docpath(dp, '..,.text'), is_ref=True, score=1.0) for doc_rep, dp in doc_index.items()
        if str(year) in doc_rep]
    return docpath_scores

class Workflow:
    def init(self, query_id=0):
        from ragpipe.config import load_config
        config = load_config('examples/billionaires.yml', show=True)
        
        data_folder = config.etc['data_folder']
        assert Path(data_folder).exists(), f'Data folder not found. Please clone github.com/ragpipe/data and point config variable etc/data_folder in startups.yml to the data folder.'
        
        md_path = f'{data_folder}/billionaires.md'
        assert Path(md_path).exists(), f'Markdownfile not found: {md_path}!'
        query_text = config.queries[query_id]
        printd(1, f'Query: {query_text}')
        return config, md_path, query_text

    def build_data_model(self, md_file):
        # read md as nodes / docs. encapsulate in desired object structure
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
            sections.append(DotDict(headerpath=headerpath, text=node.get_content()))
        D = DotDict(sections=sections)
        return D

    def run(self, respond_flag=True):
        config, md_path, query_text = self.init()

        D = self.build_data_model(md_path)
        printd(3, 'over build data model')

        from ragpipe import Retriever

        docs_retrieved = Retriever(config).eval(query_text, D)
        #print('Documents: ', docs_retrieved) #response generator

        if respond_flag:
            from ragpipe.llms import respond_to_contextual_query

            resp = respond_to_contextual_query(query_text, docs_retrieved, config.prompts['qa'], config=config) 
            print('Answer:\n', resp)
        
        return docs_retrieved



if __name__ == '__main__':
    Workflow().run()
