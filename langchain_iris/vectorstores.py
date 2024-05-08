from __future__ import annotations

import contextlib
import json
import enum
import logging
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Sequence,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

from sqlalchemy import (
    Connection,
    and_,
    asc,
    VARCHAR,
    TEXT,
    Column,
    Table,
    create_engine,
    insert,
    text,
    delete,
    Row,
)
from sqlalchemy_iris import IRISListBuild
from sqlalchemy_iris import IRISVector as IRISVectorType

from sqlalchemy.orm import Session

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

Base = declarative_base()  # type: Any


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"
    DOT_PRODUCT = "dot"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE

_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"

Base = declarative_base()


class IRISVector(VectorStore):
    _conn = None
    native_vector = False

    def __init__(
        self,
        embedding_function: Embeddings,
        dimension: int = None,
        connection_string: Optional[str] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        pre_delete_collection: bool = False,
        collection_metadata: Optional[dict] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        logger: Optional[logging.Logger] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        engine_args: Optional[dict] = None,
        connection: Optional[Connection] = None,
    ) -> None:
        self.connection_string = connection_string or "iris+emb:///"
        self.embedding_function = embedding_function
        if not dimension:
            sample_embedding = embedding_function.embed_query("Hello IRISVector!")
            dimension = len(sample_embedding)
        self.dimension = dimension
        self.collection_name = collection_name
        self.pre_delete_collection = pre_delete_collection
        self.collection_metadata = collection_metadata
        self._distance_strategy = distance_strategy
        self.logger = logger or logging.getLogger(__name__)
        self.override_relevance_score_fn = relevance_score_fn
        self.engine_args = engine_args or {}
        if connection:
            self._conn = connection 
            try:
                if connection.dialect.supports_vectors:
                    self.native_vector = True
            except:  # noqa
                pass
        else: 
            self._conn = self.connect()

        self.__post_init__()

    def __post_init__(
        self,
    ) -> None:
        """
        Initialize the store.
        """

        self.create_collection()
        self.create_vector_functions()

    def create_vector_functions(self) -> None:
        if self.native_vector:
            return
        try:
            with Session(self._conn) as session:
                session.execute(
                    text(
                        """
CREATE OR REPLACE FUNCTION langchain_l2_distance(v1 VARBINARY, v2 VARBINARY)
RETURNS NUMERIC(0,16)
LANGUAGE OBJECTSCRIPT
{
    set dims = $listlength(v1)
    set distance = 0
    for i=1:1:dims {
        set diff = $list(v1, i) - $list(v2, i)
        set distance = distance + (diff * diff)
    }

    quit $zsqr(distance)
}
"""
                    )
                )
                session.execute(
                    text(
                        """
CREATE OR REPLACE FUNCTION langchain_cosine_distance(v1 VARBINARY, v2 VARBINARY)
RETURNS NUMERIC(0,16)
LANGUAGE OBJECTSCRIPT
{
    set dims = $listlength(v1)
    set (distance, norm1, norm2, similarity) = 0

    for i=1:1:dims {
        set val1 = $list(v1, i)
        set val2 = $list(v2, i)

        set distance = distance + (val1 * val2)
        set norm1 = norm1 + (val1 * val1)
        set norm2 = norm2 + (val2 * val2)
    }

    set similarity = distance / $zsqr(norm1 * norm2)
    set similarity = $select(similarity > 1: 1, similarity < -1: -1, 1: similarity)
    quit 1 - similarity
}
"""
                    )
                )
                session.execute(
                    text(
                        """
CREATE OR REPLACE FUNCTION langchain_inner_distance(v1 VARBINARY, v2 VARBINARY)
RETURNS NUMERIC(0,16)
LANGUAGE OBJECTSCRIPT
{
    set dims = $listlength(v1)
    set distance = 0

    for i=1:1:dims {
        set val1 = $list(v1, i)
        set val2 = $list(v2, i)

        set distance = distance + (val1 * val2)
    }

    quit distance
}
"""
                    )
                )
                session.commit()
        except Exception as e:
            raise Exception(f"Failed to create vector functions: {e}") from e

    @property
    def distance_strategy(self) -> str:
        if self.native_vector:
            if self._distance_strategy == DistanceStrategy.COSINE:
                return self.table.c.embedding.cosine
            elif self._distance_strategy == DistanceStrategy.DOT_PRODUCT:
                return self.table.c.embedding.DOT_product
            # elif self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            #     return "langchain_l2_distance"
            else:
                raise ValueError(
                    f"Got unexpected value for distance: {self._distance_strategy}. "
                    f"Should be one of {', '.join([ds.value for ds in DistanceStrategy])}."
                )

        if self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return "langchain_l2_distance"
        elif self._distance_strategy == DistanceStrategy.COSINE:
            return "langchain_cosine_distance"
        elif self._distance_strategy == DistanceStrategy.DOT_PRODUCT:
            return "langchain_inner_distance"
        else:
            raise ValueError(
                f"Got unexpected value for distance: {self._distance_strategy}. "
                f"Should be one of {', '.join([ds.value for ds in DistanceStrategy])}."
            )

    def connect(self) -> Connection:
        engine = create_engine(self.connection_string, **self.engine_args)
        conn = engine.connect()
        try:
            if conn.dialect.supports_vectors:
                self.native_vector = True
        except:  # noqa
            pass
        return conn

    def __del__(self) -> None:
        if self._conn:
            self._conn.close()

    @contextlib.contextmanager
    def _make_session(self) -> Generator[Session, None, None]:
        """Create a context manager for the session, bind to _conn string."""
        yield Session(self._conn)

    @property
    def table(self) -> Table:
        return Table(
            self.collection_name,
            Base.metadata,
            Column("id", VARCHAR(40), primary_key=True, default=uuid.uuid4),
            Column(
                "embedding",
                (
                    IRISVectorType(self.dimension)
                    if self.native_vector
                    else IRISListBuild(self.dimension, float)
                ),
            ),
            Column("document", TEXT, nullable=True),
            Column("metadata", TEXT, nullable=True),
            extend_existing=True,
        )

    def create_table_if_not_exists(self) -> None:
        # Define the dynamic table
        self.table

        with self._conn.begin():
            # Create the table
            Base.metadata.create_all(self._conn)

    def create_collection(self) -> None:
        if self.pre_delete_collection:
            self.delete_collection()
        self.create_table_if_not_exists()

    def delete_collection(self) -> None:
        self.logger.debug("Trying to delete collection")
        drop_statement = text(f"DROP TABLE IF EXISTS {self.collection_name};")
        with self._conn.begin():
            self._conn.execute(drop_statement)

    @classmethod
    def get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
        if "connection_string" in kwargs and kwargs["connection_string"]:
            return kwargs["connection_string"]
        return "iris+emb:///"

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided
        # in vectorstore constructor
        if self._distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.DOT_PRODUCT:
            return self._DOT_product_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self._distance_strategy}."
                "Consider providing relevance_score_fn to IRISVector constructor."
            )

    @staticmethod
    def _cosine_relevance_score_fn(distance: float) -> float:
        """Normalize the distance to a score on a scale [0, 1]."""

        return round(1.0 - distance, 15)

    @classmethod
    def from_embeddings(
        cls: Type[IRISVector],
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> IRISVector:
        """
        Construct IRISVector wrapper from raw documents and pre-
        generated embeddings.
        """

        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        dimension = len(embeddings[0])

        store = cls(
            collection_name=collection_name,
            dimension=dimension,
            distance_strategy=distance_strategy,
            embedding_function=embedding,
            pre_delete_collection=pre_delete_collection,
            engine_args=engine_args,
            **kwargs,
        )

        store.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )
        return store

    @classmethod
    def from_texts(
        cls: Type[IRISVector],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> IRISVector:
        """
        Return VectorStore initialized from texts and embeddings.
        """

        store = cls(
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            embedding_function=embedding,
            pre_delete_collection=pre_delete_collection,
            engine_args=engine_args,
            **kwargs,
        )

        store.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)
        return store

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: Iterable[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 500,
        **kwargs: Any,
    ) -> List[str]:
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        chunks_table_data = []
        with self._conn.begin():
            for document, metadata, chunk_id, embedding in zip(
                texts, metadatas, ids, embeddings
            ):
                embedding = [float(v) for v in embedding]
                chunks_table_data.append(
                    {
                        "id": chunk_id,
                        "embedding": embedding,
                        "document": document,
                        "metadata": json.dumps(metadata),
                    }
                )

                # Execute the batch insert when the batch size is reached
                if len(chunks_table_data) == batch_size:
                    self._conn.execute(insert(self.table).values(chunks_table_data))
                    # Clear the chunks_table_data list for the next batch
                    chunks_table_data.clear()

            # Insert any remaining records that didn't make up a full batch
            if chunks_table_data:
                self._conn.execute(insert(self.table).values(chunks_table_data))

        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 500,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = self.embedding_function.embed_documents(list(texts))

        return self.add_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            batch_size=batch_size,
            **kwargs,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding_function.embed_query(text=query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding_function.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return docs

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        filter_by = True
        if filter is not None:
            filter_clauses = []
            for key, value in filter.items():
                filter_clauses.append(
                    self.table.c.metadata.like(
                        "%" + json.dumps(dict(zip((key,), (value,))))[1:-1] + "%"
                    )
                )
            filter_by = and_(*filter_clauses)

        embedding = [float(v) for v in embedding]

        # Execute the query and fetch the results
        with Session(self._conn) as session:
            results: Sequence[Row] = (
                session.query(
                    self.table,
                    (
                        self.distance_strategy(embedding).label("distance")
                        if self.native_vector
                        else self.table.c.embedding.func(
                            self.distance_strategy, embedding
                        ).label("distance")
                    ),
                )
                .filter(filter_by)
                .order_by(asc("distance"))
                .limit(k)
                .all()
            )

        documents_with_scores = [
            (
                Document(
                    page_content=result.document,
                    metadata=json.loads(result.metadata),
                ),
                (
                    round(float(result.distance), 15)
                    if self.embedding_function is not None
                    else None
                ),
            )
            for result in results
        ]
        return documents_with_scores

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
        """
        with Session(self._conn) as session:
            if ids is not None:
                self.logger.debug(
                    "Trying to delete vectors by ids (represented by the model "
                    "using the custom ids field)"
                )
                if not isinstance(ids, list) and not isinstance(ids, tuple):
                    stmt = delete(self.table).where(self.table.c.id == ids)
                else:
                    stmt = delete(self.table).where(self.table.c.id.in_(ids))
                session.execute(stmt)
            session.commit()

    def get(
        self,
        ids: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Gets the collection.

        Args:
            ids: The ids of the embeddings to get. Optional.
            limit: The number of documents to return. Optional.
        """

        # Execute the query and fetch the results
        with Session(self._conn) as session:
            results: Sequence[Row] = (
                session.query(
                    self.table.c.id,
                )
                .limit(limit)
                .all()
            )

        ids = [row.id for row in results]
        return {"ids": ids}
