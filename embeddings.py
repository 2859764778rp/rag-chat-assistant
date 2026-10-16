from __future__ import annotations

import json
import math
import os
import re
import threading
from typing import List

import numpy as np

from config import logger, EMBEDDING_MODEL, VEC_DIR


class LocalEmbeddings:

    def __init__(self):
        self._lock = threading.Lock()
        self.vocab: dict = {}
        self.idf: np.ndarray = np.zeros(0, dtype=np.float32)
        self.doc_count: int = 0
        self.df: dict = {}

    _token_re = re.compile(r"[一-鿿]|[a-zA-Z0-9]+")

    def _tokenize(self, text: str) -> List[str]:
        return self._token_re.findall((text or "").lower())

    def _l2_normalize(self, mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).astype(np.float32)

    def _fit_transform(self, texts: List[str]) -> np.ndarray:
        with self._lock:
            for t in texts:
                for tok in set(self._tokenize(t)):
                    if tok not in self.vocab:
                        self.vocab[tok] = len(self.vocab)
                    self.df[tok] = self.df.get(tok, 0) + 1
                self.doc_count += 1

            dim = len(self.vocab)
            idf = np.zeros(dim, dtype=np.float32)
            for tok, idx in self.vocab.items():
                df = self.df.get(tok, 0)
                idf[idx] = math.log((1 + self.doc_count) / (1 + df)) + 1.0
            self.idf = idf
            return self._transform(texts)

    def _transform(self, texts: List[str]) -> np.ndarray:
        dim = len(self.vocab)
        mat = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = self._tokenize(t)
            if not toks:
                continue
            for tok in toks:
                idx = self.vocab.get(tok)
                if idx is not None:
                    mat[i, idx] += 1.0
            mat[i] /= max(len(toks), 1)
            if self.idf.shape[0] == dim:
                mat[i] *= self.idf
        return self._l2_normalize(mat)

    _VOCAB_PATH = VEC_DIR / "tfidf_vocab.json"
    _IDF_PATH = VEC_DIR / "tfidf_idf.npy"

    def save_state(self):
        try:
            os.makedirs(VEC_DIR, exist_ok=True)
            with open(self._VOCAB_PATH, "w", encoding="utf-8") as f:
                json.dump({"vocab": self.vocab, "df": self.df,
                           "doc_count": self.doc_count}, f, ensure_ascii=False)
            if self.idf.shape[0]:
                np.save(self._IDF_PATH, self.idf)
        except Exception as e:
            logger.warning("embeddings save_state 失败: %s", e)

    def load_state(self):
        try:
            if self._VOCAB_PATH.exists():
                with open(self._VOCAB_PATH, "r", encoding="utf-8") as f:
                    st = json.load(f)
                self.vocab = st.get("vocab", {})
                self.df = st.get("df", {})
                self.doc_count = st.get("doc_count", 0)
                if self._IDF_PATH.exists():
                    self.idf = np.load(self._IDF_PATH).astype(np.float32)
                logger.info("embeddings: 已加载 TF-IDF 词表（%d 词，%d 文档）",
                            len(self.vocab), self.doc_count)
        except Exception as e:
            logger.warning("embeddings load_state 失败: %s", e)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._fit_transform(list(texts)).tolist()

    def embed_query(self, text: str) -> List[float]:
    
        return self._transform([text])[0].tolist()

    def to_langchain(self):
       
        return self


_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    try:
        from langchain_huggingface import HuggingFaceEmbeddings 

        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logger.info("embeddings: 使用语义向量模型 %s", EMBEDDING_MODEL)
    except Exception as e:
        _embeddings = LocalEmbeddings().to_langchain()
        _embeddings.load_state()
        logger.info("embeddings: sentence-transformers 不可用，回退本地 TF-IDF。原因: %s", e)
    return _embeddings


if __name__ == "__main__":
    emb = get_embeddings()
    docs = ["AI 职业发展前景如何", "Python 课程包含哪些内容", "今天天气怎么样"]
    vecs = emb.embed_documents(docs)
    q = emb.embed_query("人工智能就业")
    q = np.asarray(q)
    sims = np.asarray(vecs) @ q
    print("backend =", type(emb).__name__, "dim =", len(vecs[0]))
    for d, s in sorted(zip(docs, sims.tolist()), key=lambda x: -x[1]):
        print(f"  {s:.3f}  {d}")
