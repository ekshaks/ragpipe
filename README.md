# Ragpipe

Ragpipe helps you get insights from your large document repositories quickly by building fast, iteration-friendly RAG pipelines.

Ragpipe makes it easy to tweak components of your RAG pipeline so that you can iterate fast until you get desired accurate responses.

Instead of the usual `chunk-embed-match-rank` flow, Ragpipe adopts a holistic, end-to-end view of the pipeline, consisting of:

- building the data model, 
- choosing representations for document parts, 
- specifying the correct bridges among representations, 
- merging the retrieved docs across bridges,
- and using the retrieved docs to compute the query response

The `represent-bridge-merge` pattern is very powerful and allows us to build all kinds of complex retrieval engines with `retrieve-rank-rerank` patterns.

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

See the example in the `ragpipe/examples/insurance` directory.



# Dependencies

The current ragpipe version relies on LlamaIndex for parsing.
