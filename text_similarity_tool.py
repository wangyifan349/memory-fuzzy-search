#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text_similarity_tool.py

支持：
1. TF-IDF + jieba/NLTK + Cosine 相似度
2. Sentence-BERT（多语言）+ Cosine 相似度

交互式选择模式并输入查询句，返回 Top-K 最相似结果。
"""
import re
import sys
import jieba
import nltk
import numpy as np
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# SentenceTransformer 可能较大，延迟导入以加快 TF-IDF 使用场景启动
try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_SBERT = True
except Exception:
    _HAS_SBERT = False

# ---------------- NLTK 依赖 ----------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
EN_STOPWORDS = set(nltk.corpus.stopwords.words('english'))

# ---------------- 默认语料（示例，可替换） ----------------
CORPUS: List[str] = [
    "我爱你",
    "你爱我",
    "Today is a good day",
    "a good day today",
    "今天天气很好",
    "天气 今天 很 好",
    "Weather is good today"
]

# ---------------- TF-IDF 分词器 ----------------
def mixed_tokenizer(text: str, ngram_range=(1, 2)) -> List[str]:
    """
    对中英文文本分词并生成 n-gram token 列表：
    - 中文：jieba.lcut
    - 英文：nltk.word_tokenize
    - 过滤：英文停用词、标点、纯数字
    - 合并 1~2 gram，以 '_' 连接
    """
    if not isinstance(text, str):
        return []
    tokens: List[str] = []
    # 捕获连续的中文或英文序列（避免把标点等当词）
    pattern = re.compile(r'[\u4e00-\u9fa5]+|[A-Za-z]+')
    for match in pattern.findall(text):
        if re.fullmatch(r'[\u4e00-\u9fa5]+', match):
            words = jieba.lcut(match)
        else:
            # 英文片段小写并分词
            words = nltk.word_tokenize(match)
        for w in words:
            w0 = w.lower().strip()
            if not w0:
                continue
            # 过滤停用词、纯数字或全非字母数字字符
            if w0 in EN_STOPWORDS or re.fullmatch(r'\d+|\W+', w0):
                continue
            tokens.append(w0)
    # 构造 n-gram（默认 1~2）
    min_n, max_n = ngram_range
    ngrams: List[str] = []
    for n in range(min_n, max_n + 1):
        if n == 1:
            ngrams.extend(tokens)
        else:
            for i in range(len(tokens) - n + 1):
                ngrams.append('_'.join(tokens[i:i+n]))
    return ngrams

# ---------------- TF-IDF 向量化 ----------------
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda txt: mixed_tokenizer(txt, ngram_range=(1, 2)),
                                   lowercase=False, token_pattern=None)
tfidf_matrix = tfidf_vectorizer.fit_transform(CORPUS)

def tfidf_top_k(query: str, top_k: int = 3) -> List[Tuple[int, float]]:
    q_vec = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]
    top_k = max(1, min(len(scores), int(top_k)))
    idxs = np.argsort(scores)[-top_k:][::-1]
    return [(int(i), float(scores[i])) for i in idxs]

# ---------------- SBERT（延迟初始化） ----------------
SBERT_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
_sbert_model = None
_sbert_embeddings = None

def ensure_sbert_loaded():
    global _sbert_model, _sbert_embeddings
    if not _HAS_SBERT:
        raise RuntimeError("sentence-transformers 未安装，无法使用 SBERT 模式。")
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
        # 预先计算语料 embeddings（convert_to_tensor=True 可提升 semantic_search 性能）
        _sbert_embeddings = _sbert_model.encode(CORPUS, convert_to_tensor=True)

def sbert_top_k(query: str, top_k: int = 3) -> List[Tuple[int, float]]:
    ensure_sbert_loaded()
    q_emb = _sbert_model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(q_emb, _sbert_embeddings, top_k=max(1, int(top_k)))[0]
    return [(int(h['corpus_id']), float(h['score'])) for h in hits]

# ---------------- 交互逻辑 ----------------
def print_corpus():
    print("语料库：")
    for idx, s in enumerate(CORPUS):
        print(f"  [{idx}] {s}")

def main():
    print("=== 文本相似度比较工具 ===")
    print_corpus()
    print("---------------------------")
    while True:
        try:
            mode = input("请选择模式 (1) TF-IDF  (2) SBERT  (q) 退出： ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。"); break
        if mode in ('q', 'quit'):
            print("退出程序。"); break
        if mode not in ('1', '2'):
            print("无效输入，请输入 1、2 或 q。"); continue
        query = input("请输入待检索文本： ").strip()
        if not query:
            print("输入为空，请重新输入。"); continue
        try:
            top_k_input = input("返回 Top-K（默认 3）： ").strip()
            top_k = int(top_k_input) if top_k_input else 3
        except ValueError:
            print("Top-K 非法，使用默认 3。"); top_k = 3
        try:
            if mode == '1':
                results = tfidf_top_k(query, top_k=top_k)
                print("【TF-IDF Top-{} 相似句】".format(len(results)))
            else:
                results = sbert_top_k(query, top_k=top_k)
                print("【SBERT Top-{} 相似句】".format(len(results)))
            for idx, score in results:
                print(f"  [{idx}] {CORPUS[idx]}   sim={score:.4f}")
        except Exception as e:
            print("运行出错：", str(e))
        print("---------------------------")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断，程序结束。")
        sys.exit(0)
