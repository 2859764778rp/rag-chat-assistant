from __future__ import annotations

import json
import os
import threading
import uuid
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from config import VEC_DIR, RAG_TOP_K, logger
from embeddings import get_embeddings


class LocalVectorStore(VectorStore):

    def __init__(self, embeddings=None, name: str = "default",
                 persist_dir: str = str(VEC_DIR)):
        super().__init__()
        self._embed = embeddings if embeddings is not None else get_embeddings()
        self.name = name
        self.vec_path = f"{persist_dir}/{name}.npy"
        self.meta_path = f"{persist_dir}/{name}.json"
        self._lock = threading.Lock()
        self._vectors: Optional[np.ndarray] = None
        self._docs: List[dict] = []
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self._docs = json.load(f)
            if os.path.exists(self.vec_path):
                self._vectors = np.load(self.vec_path).astype(np.float32)
            
            _load_embed_state(self._embed)
            logger.info("vectorstore[%s]: 载入 %d 条文档", self.name, len(self._docs))
        except Exception as e:
            logger.warning("vectorstore[%s]: 载入失败，重新初始化。原因: %s", self.name, e)

    def _persist(self):
        os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._docs, f, ensure_ascii=False, indent=2)
        if self._vectors is not None:
            np.save(self.vec_path, self._vectors)
     
        _save_embed_state(self._embed)

    def _align_dims(self, mat: np.ndarray, dim: int) -> np.ndarray:
        if mat.shape[1] == dim:
            return mat
        if mat.shape[1] < dim:
            return np.hstack([mat, np.zeros((mat.shape[0], dim - mat.shape[1]),
                                            dtype=np.float32)])
        return mat[:, :dim]

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None,
                  ids: Optional[List[str]] = None, **kwargs: Any) -> List[str]:
        texts = list(texts)
        if not texts:
            return []
        metadatas = metadatas or [{} for _ in texts]
        if len(metadatas) != len(texts):
            metadatas = [{} for _ in texts]
        new_vecs = np.asarray(self._embed.embed_documents(texts), dtype=np.float32)

        with self._lock:
            ret_ids = []
            for i, (text, meta) in enumerate(zip(texts, metadatas)):
                doc_id = ids[i] if ids and i < len(ids) else str(uuid.uuid4())
                ret_ids.append(doc_id)
                self._docs.append({"id": doc_id, "text": text, "meta": meta})
            if self._vectors is None:
                self._vectors = new_vecs
            else:
                dim = max(self._vectors.shape[1], new_vecs.shape[1])
                self._vectors = np.vstack([self._align_dims(self._vectors, dim),
                                           self._align_dims(new_vecs, dim)])
            self._persist()
            logger.info("vectorstore[%s]: 新增 %d 条，共 %d 条",
                        self.name, len(texts), len(self._docs))
            return ret_ids

    def similarity_search(self, query: str, k: int = RAG_TOP_K,
                          **kwargs: Any) -> List[Document]:
        return [doc for doc, _score in
                self._similarity_search_with_score(query, k=k, **kwargs)]

    def _similarity_search_with_score(self, query: str, k: int = RAG_TOP_K,
                                      **kwargs: Any) -> List[Tuple[Document, float]]:
        with self._lock:
            if not self._docs or self._vectors is None:
                return []
            q = np.asarray(self._embed.embed_query(query), dtype=np.float32)
            mat = self._align_dims(self._vectors, q.shape[0])
            q = q[: mat.shape[1]]
            sims = (mat @ q).ravel()
            k = min(k, len(self._docs))
            idx = np.argsort(-sims)[:k]
            return [
                (Document(page_content=self._docs[i]["text"],
                          metadata={**self._docs[i].get("meta", {}),
                                    "store": self.name}),
                 float(sims[i]))
                for i in idx
            ]

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        if not ids:
            return False
        with self._lock:
            id_set = set(ids)
            keep = [i for i, d in enumerate(self._docs) if d["id"] not in id_set]
            removed = len(self._docs) - len(keep)
            if removed == 0:
                return False
            self._docs = [self._docs[i] for i in keep]
            self._vectors = self._vectors[keep] if keep else None
            self._persist()
            logger.info("vectorstore[%s]: 删除 %d 条，剩余 %d 条",
                        self.name, removed, len(self._docs))
            return True

    def count(self) -> int:
        return len(self._docs)

    @classmethod
    def from_texts(cls, texts: Iterable[str], embedding=None,
                   metadatas: Optional[List[dict]] = None,
                   ids: Optional[List[str]] = None, **kwargs: Any) -> "LocalVectorStore":
        name = kwargs.get("name", "default")
        persist_dir = kwargs.get("persist_dir", str(VEC_DIR))
        store = cls(embedding=embedding, name=name, persist_dir=persist_dir)
        store.add_texts(list(texts), metadatas=metadatas, ids=ids)
        return store


def _save_embed_state(embed):
    fn = getattr(embed, "save_state", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass


def _load_embed_state(embed):
    fn = getattr(embed, "load_state", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass


_stores: dict = {}
_stores_lock = threading.Lock()


def get_store(name: str = "default") -> LocalVectorStore:
    with _stores_lock:
        if name not in _stores:
            _stores[name] = LocalVectorStore(name=name)
        return _stores[name]


if __name__ == "__main__":
    s = get_store("_smoketest")
    s.add_texts(["AI 工程师的职业路径", "深度学习课程大纲", "报名费用与优惠"])
    for r in s.similarity_search("学习深度学习要多少钱", k=3):
        print(r.page_content)
