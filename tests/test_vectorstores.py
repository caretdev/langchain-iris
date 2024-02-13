import os
from typing import List
import random
import string

import pytest

from langchain.docstore.document import Document
from sqlalchemy.orm import Session

from langchain_iris import IRISVector
from langchain.embeddings.fake import DeterministicFakeEmbedding


class FakeEmbeddings(DeterministicFakeEmbedding):
    size = 200


class FakeEmbeddingsWithAdaDimension(FakeEmbeddings):
    size = 1536
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [[float(1.0)] * (self.size - 1) + [float(i)] for i in range(len(texts))]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (self.size - 1) + [float(0.0)]


@pytest.fixture(scope="class")
def connection_string(request):
    return request.config.getoption("--dburi")


@pytest.fixture(scope="function")
def collection_name(request):
    return "test_" + "".join(
        random.choices(string.ascii_lowercase + string.digits, k=10)
    )


def test_irisvector(collection_name, connection_string) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = IRISVector.from_texts(
        texts=texts,
        collection_name=collection_name,
        embedding=DeterministicFakeEmbedding(size=200),
        connection_string=connection_string,
        pre_delete_collection=True,
    )
    for doc in texts:
        output = docsearch.similarity_search(doc, k=1)
        assert output == [Document(page_content=doc)]


def test_irisvector_embeddings(collection_name, connection_string) -> None:
    """Test end to end construction with embeddings and search."""
    texts = ["foo", "bar", "baz"]
    text_embeddings = FakeEmbeddings().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = IRISVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        collection_name=collection_name,
        embedding=FakeEmbeddings(),
        connection_string=connection_string,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_irisvector_with_metadatas(collection_name, connection_string) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = IRISVector.from_texts(
        texts=texts,
        collection_name=collection_name,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        connection_string=connection_string,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_irisvector_with_metadatas_with_scores(
    collection_name, connection_string
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = IRISVector.from_texts(
        texts=texts,
        collection_name=collection_name,
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=connection_string,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_irisvector_with_filter_match(collection_name, connection_string) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = IRISVector.from_texts(
        texts=texts,
        collection_name=collection_name,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        connection_string=connection_string,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "0"})
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_irisvector_with_filter_distant_match(
    collection_name, connection_string
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = IRISVector.from_texts(
        texts=texts,
        collection_name=collection_name,
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=connection_string,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "2"})
    assert output == [
        (Document(page_content="baz", metadata={"page": "2"}), 0.001300390667138)
    ]


def test_irisvector_with_filter_no_match(collection_name, connection_string) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = IRISVector.from_texts(
        texts=texts,
        collection_name=collection_name,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        connection_string=connection_string,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "5"})
    assert output == []


def test_irisvector_delete_docs(collection_name, connection_string) -> None:
    """Add and delete documents."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = IRISVector.from_texts(
        texts=texts,
        collection_name=collection_name,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        ids=["1", "2", "3"],
        connection_string=connection_string,
        pre_delete_collection=True,
    )
    output = docsearch.get()
    assert output["ids"] == ["1", "2", "3"]

    docsearch.delete(["1", "2"])
    with docsearch._make_session() as session:
        records = list(session.query(docsearch.table).all())
        assert sorted(record.id for record in records) == ["3"]  # type: ignore

    docsearch.delete(["2", "3"])  # Should not raise on missing ids
    with docsearch._make_session() as session:
        records = list(session.query(docsearch.table).all())
        assert sorted(record.id for record in records) == []  # type: ignore


def test_irisvector_delete_docs_uuid(collection_name, connection_string) -> None:
    """Add and delete documents."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = IRISVector.from_texts(
        texts=texts,
        collection_name=collection_name,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        connection_string=connection_string,
        pre_delete_collection=True,
    )

    output = docsearch.get()
    ids = output["ids"]

    # delete as one ID
    docsearch.delete(ids[0])
    with docsearch._make_session() as session:
        records = list(session.query(docsearch.table).all())
        assert sorted(record.id for record in records) == ids[1:]  # type: ignore

    docsearch.delete(ids[0:2])
    with docsearch._make_session() as session:
        records = list(session.query(docsearch.table).all())
        assert sorted(record.id for record in records) == [ids[2]]  # type: ignore

    docsearch.delete(ids[1:])  # Should not raise on missing ids
    with docsearch._make_session() as session:
        records = list(session.query(docsearch.table).all())
        assert sorted(record.id for record in records) == []  # type: ignore


def test_irisvector_relevance_score(collection_name, connection_string) -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = IRISVector.from_texts(
        texts=texts,
        collection_name=collection_name,
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=connection_string,
        pre_delete_collection=True,
    )

    output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.999674426167506),
        (Document(page_content="baz", metadata={"page": "2"}), 0.998699609332862),
    ]


def test_irisvector_retriever_search_threshold(
    collection_name, connection_string
) -> None:
    """Test using retriever for searching with threshold."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = IRISVector.from_texts(
        texts=texts,
        collection_name=collection_name,
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=connection_string,
        pre_delete_collection=True,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.999},
    )
    output = retriever.get_relevant_documents("summer")
    assert output == [
        Document(page_content="foo", metadata={"page": "0"}),
        Document(page_content="bar", metadata={"page": "1"}),
    ]


@pytest.mark.filterwarnings("ignore::UserWarning:")
def test_irisvector_retriever_search_threshold_custom_normalization_fn(
    collection_name, connection_string
) -> None:
    """Test searching with threshold and custom normalization function"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = IRISVector.from_texts(
        texts=texts,
        collection_name=collection_name,
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=connection_string,
        pre_delete_collection=True,
        relevance_score_fn=lambda d: d * 0,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    output = retriever.get_relevant_documents("foo")
    assert output == []
