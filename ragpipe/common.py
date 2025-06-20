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
#TODO: https://github.com/rayokota/jsonata-python
def compile_jq(expression, data):
    n = 3
    l1, l2 = n, n+1

    def prepend_path(item_path_pair, edge): 
        # returns (item, edge.item_path)*

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
            elif key == '[]':
                item_path_pairs = [prepend_path(
                                    traverse(item, path[1:]), k
                                ) 
                               for k, item in obj.items()]
                res = [item for sublist in item_path_pairs for item in sublist]
                return res
            else:
                raise ValueError(f'Invalid: key = {key}, obj keys={list(obj.keys())}')
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
    try:
        items, item_paths = [list(tupleObj) for tupleObj in zip(*item_path_pairs)]
    except Exception as e:
        print(f'Error in get_fpath_items: {e}')
        breakpoint()
        print(type(item_path_pairs), len(item_path_pairs))
        raise e

    printd(3, f'get_fpath_items = {type(items)}, {len(items)}, {items[:5]}')
    #quit()
    return DotDict(els=items, paths=item_paths)

def tfm_docpath(path: 'docpath', tfm: str):
    # sections[].header -- ..,.text --> sections[].text
    # jsonata: sections.header + '%.text' -> sections.text
    #https://docs.jsonata.org/path-operators#-parent
    
    #TODO: only handles '..' now. generalize.
    sep = '/' if '/' in tfm else ','
    parts = tfm.split(sep)
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

def get_field_value_by_tfm(doc_path, tfm_path, D, extract_single=True):

    field_path = tfm_docpath(doc_path, tfm_path)
    item_dict = get_fpath_items(field_path, D)
    els = item_dict['els']
    if extract_single and len(els) == 1: els = els[0]
    return els

def get_collection_name(fpath, repname):
    return 'C_' + fpath.replace('[]', '-').replace('.', '_') + f'_{repname}' + '_C'



def generate_uuid_from_string(input_string):
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_OID, input_string))


def detect_type(item):
    from PIL import Image
    from io import BytesIO
    from pathlib import Path
    try:
        if isinstance(item, Image.Image):
            return "Image"
        if isinstance(item, Path): #only allow image files
            Image.open(item).verify()
            return "Image"

        # Try opening the item as an image
        Image.open(BytesIO(item)).verify()
        return "Image"
    except (IOError, SyntaxError, TypeError):
        try:
            # If it's not an image, check if it's text
            item.decode('utf-8')
            return "Text"
        except (UnicodeDecodeError, AttributeError):
            return "Unknown"


import time
import jsonlines
from contextlib import contextmanager


@contextmanager
def rp_timer(name, run_data):
    if len(run_data) == 0:
        run_data.update({"run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "sections": []})

    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    
    part_entry = {
        "section": name,
        "execution_time": round(elapsed_time, 6)
    }
    
    print(part_entry)  # Optional: Print to console
    run_data["sections"].append(part_entry)

def test_timer():
    log_file = "timing_log.jsonl"

    run_data = {}
    with rp_timer("Part 1", run_data):
        sum([i**2 for i in range(1000000)])
    
    with rp_timer("Part 2", run_data):
        time.sleep(1.5)
    
    with rp_timer("Part 3", run_data):
        for _ in range(5000000):
            pass
    
    # Log entire run data as a single JSONLines entry
    with jsonlines.open(log_file, mode='a') as writer:
        writer.write(run_data)

if __name__ == '__main__':
    test_timer()
