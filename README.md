# Langchain with InterSystems IRIS

[Langchain](https://github.com/langchain-ai/langchain) with support for InterSystems IRIS


## Install

```shell
pip install langchain-iris
```

## Example

```python
import os
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.fastembed import FastEmbedEmbeddings

from langchain_iris import IRISVector

loader = TextLoader("state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

CONNECTION_STRING = 'iris://_SYSTEM:SYS@localhost:1972/USER'

load_dotenv(override=True)

embeddings = OpenAIEmbeddings()

COLLECTION_NAME = "state_of_the_union_test"

db = IRISVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db.similarity_search_with_score(query)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
```