from typing import List, Any
from pydantic import BaseModel
from pathlib import Path
from .common import DotDict, detect_type, printd
from .common import get_field_value_by_tfm


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
    
    def get_file_path(self):
        assert isinstance(self.li_node, Path), f'file path not found. doc type is {type(self.li_node)}'
        return self.li_node
    
    def tfm_docpath(self, path, D, reload=True):
        from .common import tfm_docpath as tfm
        self.doc_path = tfm(self.doc_path, path)
        self.is_ref = True
        #print(self.doc_path)
        if reload: 
            self.load_docs(D)
            self.is_ref = False

    def get_field_values(self, fields, D):
        field_values = {field: get_field_value_by_tfm(self.doc_path, f'..,.{field}', D) for field in fields}
        return field_values
    
    def get_doc_rep(self, field2path, D, extract_single=True):
        res = DotDict()
        for field, path in field2path.items():
            els = get_field_value_by_tfm(self.doc_path, path, D, extract_single=extract_single)
            res[field] = els
            #print('\nget_doc_rep: ', field_path, "->", res[field])
        return res


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
            assert False, f'cannot load document {self.doc_path}'
            return False
        #return self


class ScoreNode(DocNode):
    score: float = None
    def show(self, truncate_at=100, doc_id = None):
        assert not self.is_ref, 'Load doc before display'
        doc_id = self.doc_path if doc_id is None else doc_id

        if isinstance(self.li_node, str):  #, f'li_node has type {type(self.li_node)}'
            values = ''
            if self.bridge2rank is not None:
                #values = '[' + ','.join(map(str,self.bridge2rank.values())) + ']'
                values = '[' + ','.join([f'{b}:{rank}' for b, rank in self.bridge2rank.items()]) + ']'
            text = self.li_node[:truncate_at]
            print(f' 👉 {self.score:.3f} {values} ({doc_id}) 👉 ', text)
        elif detect_type(self.li_node) == 'Image':
            path = self.li_node if isinstance(self.li_node, Path) else ''
            print(' 👉 ', self.score, doc_id, path, '\n\n')
        else:
            truncate_at = truncate_at or 400
            text = self.li_node.get_content()[:truncate_at]
            print(' 👉 ', self.score, doc_id, text, '\n\n')



class PathNode(ScoreNode): #TODO: get rid (absorb here) of ScoreNode, DocNode later
    pass

from collections.abc import MutableSequence

class PathNodes(MutableSequence):
    def __init__(self, *args, **kwargs):
        assert len(args) == 1
        self._items = args[0]  # Store items in a list

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        if not isinstance(value, PathNode):
            value = PathNode(value)
        self._items[index] = value

    def __delitem__(self, index):
        del self._items[index]

    def __len__(self):
        return len(self._items)

    def insert(self, index, value):
        if not isinstance(value, PathNode):
            value = PathNode(value)
        self._items.insert(index, value)

    def append(self, item):
        if not isinstance(item, PathNode):
            item = PathNode(item)
        self._items.append(item)

    def show(self, truncate_text_at=100, include_fields = [], D: 'DocStore'=None):
        for pos, item in enumerate(self._items):
            doc_id = None
            if len(include_fields) != 0:
                assert D is not None
                field_values = item.get_field_values(include_fields, D)
                doc_id = ' '.join([f'{field}: {value}' for field, value in field_values.items()])

            print(f'({pos}) ', end='')
            item.show(truncate_text_at, doc_id=doc_id)
