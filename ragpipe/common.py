class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)

DEFAULT_LIMIT=200 #max number of documents returned on match
    
def printd(N, text):
    if N >=3: return
    print(text)

def has_field(obj, field):
    if isinstance(obj, dict):
        return field in obj
    else:
        return hasattr(obj, field)

from importlib import import_module

def load_func(dotpath : str):
    """ load function in module.  function is right-most segment """
    #sys.path.append('/path/to/module/')
    module_, func = dotpath.rsplit(".", maxsplit=1)
    m = import_module(module_)
    return getattr(m, func)

import re
def compile_jq(expression, data):
    l1, l2 = 3, 4

    def prepend_path(item_path_pair, edge):
        #printd(l1, f'prepend_path, type={type(item_path_pair)}')
        if isinstance(item_path_pair, tuple):
            item, path = item_path_pair
            return (item, f'{edge}.{path}')
        elif isinstance(item_path_pair, list):
            return [prepend_path(ip, edge) for ip in item_path_pair]
        #elif isinstance(item_path_pair, (str,dict)):
        #    return (item_path_pair, f'{edge}')
        else:
            return (item_path_pair, f'{edge}') #catch all leaves with path=[] here
            #raise ValueError(f'prepend_path: unknown input {item_path_pair}')
        
    def traverse(obj, path):
        printd(l1, f'cjq: path = {path}')
        if len(path) == 0:
            return obj
        key = path[0]
        printd(l1, f'cjq> key = {key}, obj={type(obj)}')
        '''
        if isinstance(obj, list):
            printd(l1, f'obj = {obj[:2]}')
        if isinstance(obj, str):
            printd(l1, f'obj = {obj[:20]}')  
        '''


        if isinstance(obj, dict):
            if key in obj:
                return prepend_path(traverse(obj[key], path[1:]), key)
            else:
                raise ValueError(f'Invalid: key = {key}, obj={obj}')
        elif isinstance(obj, list):
            if key == '[]':
                item_path_pairs = [prepend_path(
                                    traverse(item, path[1:]), pos
                                ) 
                               for pos, item in enumerate(obj)]
                res = item_path_pairs
            else:
                key = int(key) 
                if key >= len(obj): raise KeyError(f'position {key} not found. (len(obj)={len(obj)}) (path = {path})')
                obj_ = obj[key]
                res = prepend_path(traverse(obj_, path[1:]), key)
            #TODO:  add filter here
            return res
        else:
            return obj

    def parse(expression):
        expression_array = re.findall(r'\w+|\[\]', expression) #split f.e[].d
        printd(l1, f'cjq: {expression_array}')
        return expression_array

    path = parse(expression)
    ret = traverse(data, path)
    if isinstance(ret, tuple):
        ret = [ret]
    assert isinstance(ret, list) #always returns lists of items
    printd(l2, f'cjq: traverse return: {ret}')
    return ret


def get_fpath_items(fpath, D, docpath_pre_filter=set()):
    item_path_pairs = compile_jq(fpath, D)
    if len(docpath_pre_filter) != 0:
        item_path_pairs = [(item, item_path) for item, item_path in item_path_pairs 
                       if item_path in docpath_pre_filter]
    #print(items[:5])
    items, item_paths = [list(tupleObj) for tupleObj in zip(*item_path_pairs)]

    printd(3, f'get_fpath_items = {type(items)}, {len(items)}')
    return DotDict(els=items, paths=item_paths)

def tfm_docpath(path: 'docpath', tfm: str):
    # sections[].header -- ..,.text --> sections[].text
    #TODO: only handles '..' now. generalize.
    parts = tfm.split(',')
    opath = path
    for p in parts:
        match p:
            case '..':
                pparts = opath.split('.') #sections[], header
                if len(pparts) > 1:
                    opath = '.'.join(pparts[:-1])
                else:
                    raise ValueError('Invalid op: {p} on {opath} ')
            case _:
                opath = opath + p
    return opath

def get_collection_name(fpath, repname):
    return 'C_' + fpath.replace('[]', '-').replace('.', '_') + f'_{repname}' + '_C'



def generate_uuid_from_string(input_string):
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_OID, input_string))


def detect_type(item):
    from PIL import Image
    from io import BytesIO
    try:
        if isinstance(item, Image.Image):
            return "Image"
        # Try opening the item as an image
        Image.open(BytesIO(item)).verify()
        return "Image"
    except (IOError, SyntaxError):
        try:
            # If it's not an image, check if it's text
            item.decode('utf-8')
            return "Text"
        except (UnicodeDecodeError, AttributeError):
            return "Unknown"