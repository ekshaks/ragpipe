
## Ragpipe

Ragpipe helps you build tools to get insights from your large document repositories quickly by building fast, iteration-friendly, RAG pipelines.

Ragpipe makes it easy to tweak components of your RAG pipeline so that you can iterate fast until you get desired accurate responses.

Instead of the usual `chunk-embed-match-rank` flow, Ragpipe adopts a holistic, end-to-end view of the pipeline, consisting of:

- building the data model, 
- choosing representations for document parts, 
- specifying the correct bridges among representations, 
- merging the retrieved docs across bridges,
- and using the retrieved docs to compute the query response

The `represent-bridge-merge` pattern is very powerful and allows us to build all kinds of complex retrieval engines with `retrieve-rank-rerank` patterns.

## Installation

```
pip install ragpipe
```

## Key Ideas

**Representations**. Choose the query/document fields as well as how to represent each chosen query / document field to aid similarity/relevance computation (*bridges*) over the entire document repository. Representations can be text strings, dense/sparse vector embeddings or arbitrary data objects, and help *bridge* the gap between the query and the documents.

**Bridges**. Choose a *pair* of query and document representation to *bridge*. A bridge serves as a relevance indicator: one of the several criteria for identifying the relevant documents for a query. In practice, several bridges together determine the degree to which a document is relevant to a query. Computing each bridge creates a unique ranked list of documents.

**Merges**. Specify how to combine the bridges, e.g., combine multiple ranked list of documents into a single ranked list.

**Data Model**. A hierarchical data structure that consists of all the (nested) documents. The data model is created from the original document files and is retained over the entire pipeline. We compute representations for arbitrary nested fields of the data, without flattening the data tree.

To query over a data repository, 

- we compute the data model over the original data repository 
- specify the document fields and the (multiple) representations to be computed for each field
- specify which representations to compute for query
- specify bridges: which pair of query and doc field representation should be matched
- merges: how to combine multiple bridges, sequentially or in parallel, to yield a curated ranked list of relevant documents.
- gen-response: how to generate response to the query using the relevant document list and a large language model.


## Quick Start

See the example in the `ragpipe/examples/insurance` directory. The `main` function from `insurance.py` is inlined below.

```python
def main(respond_flag=False):
    config = load_config('examples/insurance/insurance.yml', show=True) #L1
    
    D = build_data_model('examples/data/insurance/niva-short.mmd') #L2

    query_text = config.queries[1] #L3

    from ragpipe import Retriever

    docs_retrieved = Retriever(config).eval(query_text, D) #L4

    for doc in docs_retrieved: doc.show() #l5

    if respond_flag:
        return respond(query_text, docs_retrieved, config.prompts['qa2']) #L6
    else:
        return docs_retrieved

```

In `main`, we implement the following steps:

- `#L1` read the config file [`insurance.yml`](https://github.com/ekshaks/ragpipe/examples/insurance/insurance.yml) which specifies representations, bridges and merges. 
    - go through the `insurance.yml` file to understand the definitions.
- `#L2` read the data files (`niva-short.mmd`) to build a doc model `D` with following nested fields 
    -  `query.text` which contains the query string 
    - `.sections[].node` which contains the document snippets
- `#L4` all the heavy lifting happens in `ragpipe.Flow` function based on definitions from `insurance.yml`
    - compute `#dense` representations for `query.text` and `.sections[].node` fields using `colbert-ir/colbertv2.0`
    - rank documents according to the bridge `b1` across the two representations `query.text#dense`, `.sections[].node#dense`
    - compute a merge `c1` to obtain the final ranked list. In this case, the merge is trivial since only a single bridge is defined. 
- `#L6` send the retrieved documents as context to the LLM, along with a QA prompt defined in `insurance.yml`; generate the final cohesive response.

## Key Dependencies

Ragpipe relies on 
- `rank_bm25`: for BM25 based retrieval
- `fastembed`: dense and sparse embeddings
- `chromadb`, `qdrant-client`: vector databases (more coming..)
- `litellm`: interact with LLM APIs
- `jinja2`: prompt formatting
- `LlamaIndex`: for parsing documents

To keep Ragpipe lean, we restrict the key dependencies to only a few.

Additional encoders, vector databases can be added as plugins to `ext/libs` and loaded dynamically.