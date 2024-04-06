from typing import List, Any
from pydantic import BaseModel
from .common import printd

class DocNode(BaseModel):
    type: str = 'text' #text, image, audio
    #content in memory
    content: str = None #text parts, b64 encoded
    li_node: Any = None

    #ref to content
    file_name: str = None #stored in file, not loaded yet
    doc_path: str = None #full path in doc hierarchy (single or multi-)

    is_ref: bool = True #by default - doc is a ref (not loaded)
    
    def __init__(self, **data):
        super().__init__(**data)
        #  post-initialization tasks 
        if 'li_node' in data or 'file_name' in data:
            self.is_ref = False

    def get_text_content(self):
        if self.li_node is not None:
            try:
                text = self.li_node.node.text
            except Exception:
                text = self.li_node.text
            #printd(2, f'get_text_content -- returning text: {text}')
            return text
        else:
            return self.content
    
    def load_docs(self, D):
        if not self.is_ref: 
            printd(1, 'DocNode:load_docs -- already loaded docs')
            return #already loaded
        from .common import get_fpath_items
        self.li_node = get_fpath_items(self.doc_path, D).els[0]
        self.is_ref = False
        #return self




class ScoreNode(DocNode):
    score: float = None

    def show_li_node(self):
        assert not self.is_ref, 'Load doc before display'
        node_ = self.li_node
        print(' 👉 ', self.score, node_.get_content()[:400], '\n\n')