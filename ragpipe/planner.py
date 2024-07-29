from .common import printd
from ragpipe.llms import groq_llm


decompose_prompt = '''
Decompose the following complex query into "atomic queries":
- each atomic query is a simple query, answerable from a small document part.
- the sequence of atomic queries will answer the original query.
- an atomic query Q_i may refer to the result of a previous query Q_j as "#j", starting from "#1".
- do not include any atomic queries not implied by the original query.
- output the decomposed query in the format below.

# Original query
{query}

# Decomposed query
<query> ...  ... </query>
<query> ... #1 .. </query>
<query> ... #2 .. </query>

'''

def parse_decomposed_result(result):
    queries = []
    lines = [l.strip() for l in result.split('\n')]
    printd(3, lines)
    for line in lines :
        if line.startswith("<query>") and line.endswith("</query>"):
            queries.append(line[7:-8].strip())  # Remove "<query>" and "</query>" tags
    return queries

def decompose_query(query, config):
    prompt = decompose_prompt.format(query=query)
    resp = groq_llm(prompt, model=config.llm_models['decomposer'])
    queries = parse_decomposed_result(resp)
    return queries

from .bridge import bridge_query_doc
from .llms import respond_to_contextual_query
import re

def extract_answer(result, query_text, config):
    for i in range(2):
        match = re.search(r'<answer>(.*?)</answer>', result, re.DOTALL)
        if match:
            res = match.group(1)
            break
        else:
            res = None
            printd(2, f'Rewriting trial {i}')
            prompt = config.prompts['rewrite_in_format'].format(answer=result, query=query_text)
            res = groq_llm(prompt, model=config.llm_models['formatter'])
            #result = respond_to_contextual_query(query_text, [result],  )
    return res

def link_result_query(old_results, query):
    matches = re.findall(r"#(\d+)", query) 
    nums = [int(match) for match in matches]
    if len(nums) == 0: 
        return query
    assert min(nums) >= 1
    assert max(nums) <= len(old_results), f'{nums}, {old_results}'
    substitutions = {f'#{num}': old_results[num-1]['answer'] for num in nums}

    for numtext, text in substitutions.items():
        query = query.replace(numtext, str(text))
    return query

def resolve_query(query, D, config, human_input=False):

    printd(2, 'decomposing query...')
    queries = decompose_query(query, config)
    print('decomposed query:\n', queries)
    index = 0
    query_text = queries[0]
    results = []

    while True:
        if index >= len(queries): break
        query_text = link_result_query(results[:index], queries[index])
        printd(2, f'resolve_query - next query:  {query_text}')
        if human_input: 
            query_text_h = input('update next query?:')
            if query_text_h: query_text = query_text_h.strip()
    
        docs_retrieved = bridge_query_doc(query_text, D, config)
        printd(2, 'resolve_query - docs retrieved:')
        doc_paths = []
        for d in docs_retrieved: 
            doc_paths.append(d.doc_path)
            d.show()

        result = respond_to_contextual_query(query_text, docs_retrieved, config.prompts['qa'],
                                            model = config.llm_models['answer_gen']) 
        answer = extract_answer(result, query_text, config)
        print(result, '\n', f'Answer: {answer}')

        if human_input:
            answer_h = input('update answer?:')
            if answer_h: answer = answer_h.strip()

        results.append(dict(result=result, answer=answer, query=queries[index], doc_paths=doc_paths))
        index += 1

    return results





if __name__ == '__main__':
    pass