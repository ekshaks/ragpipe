
<h1 align="center" >ragpipe</h1>
<p align="center">
    <img src="docs/src/assets/ragpipe.jpeg" width="30%" alt="Ragpipe Logo">
</p>

**Ragpipe**: Iterate fast on your RAG pipelines.

<h3 align="center">
  <a href="[https://ragpipe.github.io/](https://ragpipe.github).io">Docs</a> •
 <a href="https://discord.com/invite/ATWd8A5cEh">Discord</a> 
</h3>


## Introduction

Ragpipe helps you extract insights from large document repositories *quickly*. 

Ragpipe is lean and nimble. Makes it easy to iterate fast, tweak components of your RAG pipeline until you get desired responses.

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

The `represent-bridge-merge` pattern is very powerful and allows us to build and iterate over all kinds of complex retrieval pipelines, including those based on the traditional `retrieve-rank-rerank` pattern and more recent advanced RAG patterns.


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

See the example in the `examples/insurance` directory.


# Key Dependencies

Ragpipe relies on 
- `LlamaIndex`: for parsing markdown documents
- `rank_bm25`: for BM25 based retrieval
- `fastembed`: dense and sparse embeddings
- `litellm`: interact with LLM APIs
