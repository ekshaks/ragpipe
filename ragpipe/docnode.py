from typing import List, Any
from pydantic import BaseModel
from .common import detect_type, printd

class DocNode(BaseModel):
    type: str = 'text' #text, image, audio
    #content in memory
    #content: str = None #text parts, b64 encoded
    li_node: Any = None #using this field for all content for now

    #ref to content
    file_name: str = None #stored in file, not loaded yet
    doc_path: str = None #full path in doc hierarchy (single or multi-)
    doc_path_index: int = None #temporary index into doc_paths list

    is_ref: bool = True #by default - doc is a ref (not loaded)
    bridge2rank: dict = None #individual bridge ranks before fusion
    
    def __init__(self, **data):
        super().__init__(**data)
        #  post-initialization tasks 
        if 'li_node' in data or 'file_name' in data:
            self.is_ref = False

    def get_text_content(self):
        if self.li_node is not None:
            if isinstance(self.li_node, str):
                return self.li_node
            
            try:
                text = self.li_node.node.text
            except Exception:
                text = self.li_node.text
            #printd(2, f'get_text_content -- returning text: {text}')
            return text
        else:
            return self.li_node
    
    def tfm_docpath(self, path, D, reload=True):
        from .common import tfm_docpath as tfm
        self.doc_path = tfm(self.doc_path, path)
        self.is_ref = True
        #print(self.doc_path)
        if reload: 
            self.load_docs(D)
            self.is_ref = False

    def load_docs(self, D):
        if not self.is_ref: 
            printd(3, 'DocNode:load_docs -- already loaded docs')
            return True #already loaded
        
        printd(3, f'loading docs...{self.doc_path}')
        from .common import get_fpath_items
        try:
            self.li_node = get_fpath_items(self.doc_path, D).els[0] #
            self.is_ref = False
            return True
        except Exception as e:
            printd(2, f'load:docs -- cannot load {self.doc_path}. {e}')
            return False
        #return self




class ScoreNode(DocNode):
    score: float = None
    def show(self, truncate_at=None):
        assert not self.is_ref, 'Load doc before display'
        if isinstance(self.li_node, str):  #, f'li_node has type {type(self.li_node)}'
            values = ''
            if self.bridge2rank is not None:
                #values = '[' + ','.join(map(str,self.bridge2rank.values())) + ']'
                values = '[' + ','.join([f'{b}:{rank}' for b, rank in self.bridge2rank.items()]) + ']'
            text = self.li_node[:truncate_at]
            print(f' 👉 {self.score:.3f} {values} ({self.doc_path}) 👉 ', text)
        elif detect_type(self.li_node) == 'Image':
            print(' 👉 ', self.score, self.doc_path, '\n\n')
        else:
            truncate_at = truncate_at or 400
            text = self.li_node.get_content()[:truncate_at]
            print(' 👉 ', self.score, text, '\n\n')
