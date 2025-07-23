#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text_similarity_tool.py

一个脚本内同时支持：
1. TF-IDF + jieba/​NLTK + Cosine 相似度
2. Sentence-BERT（多语言）+ Cosine 相似度

交互式选择模式并输入查询句，返回 Top-K 最相似结果。
"""

import re
import sys
import jieba
import nltk
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------
# 1. NLTK 依赖准备（仅对 TF-IDF 模式需要下载）
# ---------------------------------------------------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
EN_STOPWORDS = set(nltk.corpus.stopwords.words('english'))

# ---------------------------------------------------------------------
# 2. 预置语料库（可自行修改）
# ---------------------------------------------------------------------
CORPUS = [
    "我爱你",
    "你爱我",
    "Today is a good day",
    "a good day today",
    "今天天气很好",
    "天气 今天 很 好",
    "Weather is good today"
]

# ---------------------------------------------------------------------
# 3. TF-IDF 模式工具函数
# ---------------------------------------------------------------------
def mixed_tokenizer(text: str, ngram_range=(1, 2)):
    """
    对中英文文本分词并生成 n-gram token 列表：
    - 中文：jieba.lcut
    - 英文：nltk.word_tokenize
    - 过滤：英文停用词、标点、纯数字
    - 合并 1~2 gram，以 '_' 连接多词 gram
    """
    tokens = []
    pattern = re.compile(r'[\u4e00-\u9fa5]+|[A-Za-z]+')
    for match in pattern.findall(text):
        if re.fullmatch(r'[\u4e00-\u9fa5]+', match):
            words = jieba.lcut(match)           # 中文分词
        else:
            words = nltk.word_tokenize(match)   # 英文分词
        for w in words:
            w0 = w.lower().strip()
            if not w0 or w0 in EN_STOPWORDS or re.fullmatch(r'\d+|\W+', w0):
                continue
            tokens.append(w0)
    # 构造 n-gram
    min_n, max_n = ngram_range
    ngrams = []
    for n in range(min_n, max_n + 1):
        if n == 1:
            ngrams.extend(tokens)
        else:
            for i in range(len(tokens) - n + 1):
                ngrams.append('_'.join(tokens[i:i+n]))
    return ngrams

# 构建 TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    tokenizer=lambda txt: mixed_tokenizer(txt, ngram_range=(1, 2)),
    lowercase=False,
    token_pattern=None
)
# 训练 TF-IDF 矩阵
tfidf_matrix = tfidf_vectorizer.fit_transform(CORPUS)

def tfidf_top_k(query: str, top_k: int = 3):
    """
    对 query 进行 TF-IDF 编码并计算与 CORPUS 的余弦相似度，
    返回 Top-K 最相似结果 [(idx, score), ...]
    """
    q_vec = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]
    topk = np.argsort(scores)[-top_k:][::-1]
    return [(int(i), float(scores[i])) for i in topk]

# ---------------------------------------------------------------------
# 4. SBERT 模式工具函数
# ---------------------------------------------------------------------
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
sbert_model = SentenceTransformer(MODEL_NAME)
# 一次性生成语料 embeddings
sbert_embeddings = sbert_model.encode(CORPUS, convert_to_tensor=True)

def sbert_top_k(query: str, top_k: int = 3):
    """
    对 query 进行 SBERT 编码并计算与 CORPUS 的余弦相似度，
    返回 Top-K 最相似结果 [(idx, score), ...]
    """
    q_emb = sbert_model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(q_emb, sbert_embeddings, top_k=top_k)[0]
    return [(h['corpus_id'], float(h['score'])) for h in hits]

# ---------------------------------------------------------------------
# 5. 交互式主流程
# ---------------------------------------------------------------------
def main():
    print("=== 文本相似度比较工具 ===")
    print("语料库示例：")
    for idx, sent in enumerate(CORPUS):
        print(f"  [{idx}] {sent}")
    print("---------------------------")

    while True:
        mode = input("请选择模式 (1) TF-IDF  (2) SBERT  (q) 退出： ").strip()
        if mode.lower() in ('q', 'quit'):
            print("退出程序。")
            break
        if mode not in ('1', '2'):
            print("无效输入，请输入 1、2 或 q。")
            continue

        query = input("请输入待检索文本： ").strip()
        if not query:
            print("输入为空，请重新输入。")
            continue

        if mode == '1':
            results = tfidf_top_k(query, top_k=3)
            print("【TF-IDF Top-3 相似句】")
        else:
            results = sbert_top_k(query, top_k=3)
            print("【SBERT Top-3 相似句】")

        for idx, score in results:
            print(f"  [{idx}] {CORPUS[idx]}   sim={score:.4f}")
        print("---------------------------")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断，程序结束。")
        sys.exit(0)

