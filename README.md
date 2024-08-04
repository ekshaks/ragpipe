
<h1 align="center" >ragpipe</h1>
<p align="center">
    <img src="docs/src/assets/ragpipe.jpeg" width="30%" alt="Ragpipe Logo">
</p>


<h3 align="center">
    Ragpipe: Iterate fast on your RAG pipelines.
    <br><br>
  <a href="https://ragpipe.github.io/">Docs</a> •
  <a href="examples/">Examples</a> •
 <a href="https://discord.com/invite/ATWd8A5cEh">Discord</a> 
</h3>


## Introduction

Ragpipe helps you extract insights from large document repositories *quickly*. 

Ragpipe is lean and nimble. Makes it easy to iterate fast, tweak components of your RAG pipeline until you get desired responses.

Yet another RAG framework? Although popular RAG frameworks make it easy to setup RAG pipelines, they lack primitives that enable you to iterate and get to desired responses quickly. 

Watch a quick [video intro](https://www.youtube.com/playlist?list=PLLPfjV1xMkS1k9J7q2v3eQ2U-At6p3evM).

*Note: Under active development. Expect breaking changes.*

---

Instead of the usual `chunk-embed-match-rank` flow, Ragpipe adopts a holistic, end-to-end view of the pipeline:

- build a hierachical **document model**, 
- **decompose** a complex query into sub-queries 
- **resolve** sub-queries and obtain responses
- **aggregate** the query responses.

How do we resolve each sub-query?
- choose **representations** for document parts relevant to a sub-query, 
- specify the **bridges** among those representations, 
- **merge** the retrieved docs across bridges to setup a context,
- present the query and context to a language model to compute the final response

The `represent-bridge-merge` pattern is very powerful and allows us to build and iterate over all kinds of complex retrieval pipelines, including those based on the traditional `retrieve-rank-rerank` pattern and more recent advanced RAG patterns. Evals can be attached to `bridge` or `merge` nodes to verify intermediate results.


## Installation

Using `pip`.
```bash
pip install ragpipe
```

Clone and install dependencies (recommended).
```bash
git clone https://github.com/ekshaks/ragpipe; cd ragpipe
#install poetry
curl -sSL https://install.python-poetry.org | python3 - 
#install ragpipe dependencies
poetry install 
```

Alternatively you could use conda to install all necessary dependencies.
```bash
git clone https://github.com/ekshaks/ragpipe; cd ragpipe
#creating a new environment with python 3.10
conda create -n ragpipe python=3.10
#activating the environment
conda activate ragpipe
#install ragpipe dependencies
pip install -r requirements.txt
```

Note: For CUDA support on Windows/Linux you might need to install PyTorch with CUDA compiled.
For instructions follow https://pytorch.org/get-started/locally/

## Key Ideas

**Representations**. Choose the query/document fields as well as how to represent each chosen query / document field to aid similarity/relevance computation (*bridges*) over the entire document repository. Representations can be text strings, dense/sparse vector embeddings or arbitrary data objects, and help *bridge* the gap between the query and the documents.

**Bridges**. Choose a *pair* of query and document representation to *bridge*. A bridge serves as a relevance indicator: one of the several criteria for identifying the relevant documents for a query. In practice, several bridges together determine the degree to which a document is relevant to a query. A bridge is a ranker and top-k selector, rolled into one. Computing each bridge creates a unique ranked list of documents with respect to the relevance criteria.

**Merges**. Specify how to combine the bridges, e.g., combine multiple ranked list of documents into a single ranked list using rank fusion.

**Data Model**. A hierarchical data structure that consists of all the (nested) documents. The data model is created from the original document files and is retained over the entire pipeline. We compute representations for arbitrary nested fields of the data, without flattening the data tree.

## Querying with Ragpipe

To query over a data repository, 

1. Build a hierachical data model over your data repositories, e.g., `{"documents" : [{"text": ...}, ...]}`. 

2. In `config.yml`:

- Specify which document fields will be represented and how.
- Specify which representations to compute for the query.
- Specify `bridges`: which pair of query and doc field representation should be matched to find relevant documents.
- Specify `merges`: how to combine multiple bridges, sequentially or in parallel, to yield the final ranked list of relevant documents.


3. Specify how to generate response to the query using the above ranked document list and a large language model.
4. Iterate by making quick changes to (1), (2) or (3).

## Quick Start

Examples are in the [examples](examples) directory.

For instance, run [`examples/insurance`](examples/insurance).
```
examples/insurance/
|
|-- insurance.py
|-- insurance.yml
```

```bash 
python -m examples.insurance.insurance
```

The default LLM is [Groq](https://groq.com/). Please set GROQ_API_KEY in `.env`. Alternatively, openai LLMs (set `OPENAI_API_KEY`) and ollama based local LLMs (`ollama/..` or `local/..`) are also supported.

## API Usage

Embed ragpipe into your Agentic query resolvers by delegating fine-grained retrieval to ragpipe.

```python

def rag():
    from ragpipe.config import load_config
    config = load_config('examples/<project>/config.yml', show=True) #see examples/*/*.yml

    D = build_data_model(config) #D.query_text is the query, D.docs.<> contain documents

    from ragpipe.bridge import bridge_query_doc
    docs_retrieved = bridge_query_doc(D.query_text, D, config)
    for doc in docs_retrieved: doc.show()


    from ragpipe.llms import respond_to_contextual_query as respond
    result = respond(query_text, docs_retrieved, config.prompts['qa'], config.llm_models['default']) 
    
    print(f'\nQuery: {query_text}')
    print('\nGenerated answer: ', result)
```



## Key Dependencies

Ragpipe relies on 
- `rank_bm25`: for BM25 based retrieval
- `fastembed`: dense and sparse embeddings
- `chromadb`: default vector database (more coming..)
- `litellm`: interact with LLM APIs
- `jinja2`: prompt formatting
- `LlamaIndex`: for parsing documents


## Contribution

Ragpipe is open-source and under active development. We welcome contributions:
- Try out ragpipe on queries over your data. Open an issue or send a pull request.
- Join us as an early contributor to build a new, powerful RAG framework.
- Stuck on a RAG problem without progress? Share with us, iterate and overcome blockers.


Join discussion on our [Discord](https://discord.com/invite/ATWd8A5cEh) channel.


## Read More

- [Why your GPT + Vector Search RAG demo won't make it to production?](https://offnote.substack.com/p/llm-ir-1-why-your-gpt-vector-search)
- [RAG++: Bridging the Query - Doc Gap](https://offnote.substack.com/p/llm-ir-2-rag-from-scratch-bridging)
- [Lessons Building an Enterprise RAG Product](https://offnote.substack.com/p/lessons-building-an-enterprise-genai)