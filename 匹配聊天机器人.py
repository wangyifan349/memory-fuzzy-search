#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#pip install jieba scikit-learn faiss-cpu
import sys
import os
import json
import hashlib
import math
import jieba
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 尝试导入 faiss（兼容 cpu/gpu 包名）
try:
    import faiss
except Exception as e:
    raise ImportError("请先安装 faiss-cpu（或 faiss），例如：pip install faiss-cpu") from e

# -------------------- 工具函数 --------------------

def lcs_length(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def lcs_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    avg_len = (len(a) + len(b)) / 2.0
    if avg_len == 0:
        return 0.0
    return lcs_length(a, b) / avg_len

def jieba_tokenize(text: str) -> str:
    # 返回以空格分隔的 token 字符串，方便给 sklearn 的 TfidfVectorizer 使用
    return " ".join(jieba.lcut(text))
def md5_hash(question: str, answer: str) -> str:
    return hashlib.md5((question + answer).encode('utf-8')).hexdigest()
# -------------------- QASystem with TF-IDF + FAISS --------------------
class QASystem:
    def __init__(self, weight_tfidf=0.2, weight_lcs=0.8, preload=None):
        self.qa_list = []            # 每项：{'question','answer','q_text_tokenized'}
        self.qa_hash_set = set()
        self.history = []
        self.weight_tfidf = weight_tfidf
        self.weight_lcs = weight_lcs

        # TF-IDF 和 FAISS
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")  # 我们传入已经用空格分词的字符串
        self.tfidf_matrix = None     # np.array shape (n, d)
        self.index = None            # faiss index
        if preload:
            self.add_qa_list(preload, rebuild_index=True)
    def add_qa_list(self, qa_list, rebuild_index=True):
        added = 0
        for qa in qa_list:
            q = qa.get('question', '').strip().replace('\n', ' ')
            a = qa.get('answer', '').strip().replace('\n', ' ')
            if q == '' and a == '':
                continue
            h = md5_hash(q, a)
            if h in self.qa_hash_set:
                continue
            q_tokenized = jieba_tokenize(q)
            self.qa_list.append({'question': q, 'answer': a, 'q_tokenized': q_tokenized})
            self.qa_hash_set.add(h)
            added += 1
        if added > 0 and rebuild_index:
            self._rebuild_tfidf_and_faiss()
        return added
    def _rebuild_tfidf_and_faiss(self):
        # 生成 TF-IDF 矩阵（稠密）
        corpus = [item['q_tokenized'] for item in self.qa_list]
        # 重新拟合 vectorizer（简单实现）
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        tfidf_sparse = self.vectorizer.fit_transform(corpus)  # shape (n, d) sparse
        tfidf = tfidf_sparse.toarray().astype('float32')      # dense float32 for faiss
        self.tfidf_matrix = tfidf
        # 构建 faiss 索引（使用 Inner Product on normalized vectors for cosine）
        # 先 L2 归一化每行 -> cosine 等于 inner product
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tfidf_normed = tfidf / norms
        dim = tfidf_normed.shape[1]
        if self.index is None:
            # 使用 IndexFlatIP 做精确搜索（小数据集合适）
            self.index = faiss.IndexFlatIP(dim)
        else:
            # 如果已有索引，先删除并重建（简单方式）
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(tfidf_normed.astype('float32'))  # 添加向量
        # store normalized matrix for scoring convenience
        self._tfidf_normed = tfidf_normed
    def _query_tfidf_scores(self, query: str, top_k=5):
        if self.tfidf_matrix is None or len(self.qa_list) == 0:
            return [], []
        q_tok = jieba_tokenize(query)
        q_vec_sparse = self.vectorizer.transform([q_tok])
        q_vec = q_vec_sparse.toarray().astype('float32')
        # 归一化
        norm = np.linalg.norm(q_vec)
        if norm == 0:
            q_normed = q_vec
        else:
            q_normed = q_vec / norm
        # 使用 faiss 搜索 inner product（等价 cosine）
        D, I = self.index.search(q_normed, top_k)  # D: scores, I: indices
        scores = D[0].tolist()
        indices = I[0].tolist()
        return indices, scores
    def find_best_match(self, user_question, top_k=10, threshold=0.3):
        # 先用 tfidf/faiss 得到候选
        indices, tfidf_scores = self._query_tfidf_scores(user_question, top_k=top_k)
        best_score = -1.0
        best_answer = "抱歉，我暂时无法回答您的问题。"
        best_details = (0.0, 0.0, -1)  # combined, tfidf, lcs, idx
        # 如果没有候选（如空索引），遍历全部
        candidate_idxs = [i for i in indices if i != -1] if indices else list(range(len(self.qa_list)))
        if not candidate_idxs:
            candidate_idxs = list(range(len(self.qa_list)))
        for pos, idx in enumerate(candidate_idxs):
            qa = self.qa_list[idx]
            tfidf_score = tfidf_scores[pos] if pos < len(tfidf_scores) else 0.0
            lcs_sim = lcs_similarity(user_question, qa['question'])
            combined = self.weight_tfidf * tfidf_score + self.weight_lcs * lcs_sim
            if combined > best_score:
                best_score = combined
                best_answer = qa['answer']
                best_details = (combined, tfidf_score, lcs_sim, idx)
        if best_score >= threshold:
            return best_answer, best_score, best_details
        else:
            return "抱歉，我暂时无法回答您的问题。", best_score, best_details
    def ask(self, user_question):
        answer, score, details = self.find_best_match(user_question)
        self.history.append({
            'question': user_question,
            'answer': answer,
            'score': score,
            'tfidf': details[1] if len(details) > 1 else 0.0,
            'lcs': details[2] if len(details) > 2 else 0.0
        })
        return answer, score, details
# -------------------- JSON 加载与命令行 --------------------
def load_qa_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and 'qa' in data and isinstance(data['qa'], list):
                return data['qa']
            # 尝试从 dict 中抽取 question/answer
            if isinstance(data, dict):
                possible = []
                for v in data.values():
                    if isinstance(v, dict) and 'question' in v and 'answer' in v:
                        possible.append({'question': v['question'], 'answer': v['answer']})
                if possible:
                    return possible
            raise ValueError("JSON 格式不受支持（需要 list 或 包含 'qa' 键）。")
    except Exception as e:
        print(f"[错误] 加载文件 '{filepath}' 失败：{e}")
        return None
def print_help():
    print("""
可用命令：
  load <file1> [file2 ...]   加载一个或多个 JSON QA 文件
  count                      显示已加载的 QA 对数
  ask <your question>        向系统提问（或直接输入问题并回车）
  history                    显示问答历史
  save_history <file>        将问答历史保存为 JSON 文件
  help                       显示此帮助
  exit                       退出
""".strip())
def main():
    preload = [
        {'question': '如何重置密码？', 'answer': '请在设置页面点击“重置密码”，然后按照提示操作。'},
        {'question': '如何注册账号？', 'answer': '点击首页的注册按钮并填写注册信息即可。'},
        {'question': '退款政策是什么？', 'answer': '请参见我们的退款政策页面，通常在30天内可申请退款。'},
    ]
    qa_system = QASystem(weight_tfidf=0.2, weight_lcs=0.8, preload=preload)
    print("命令行问答系统（TF-IDF + FAISS 0.2 & LCS 0.8）——已加载预置数据")
    print("提示：确保已安装依赖：jieba scikit-learn faiss-cpu")
    print_help()
    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if cmd == "":
            continue
        parts = cmd.split()
        cmd0 = parts[0].lower()

        if cmd0 == 'load':
            if len(parts) < 2:
                print("请提供至少一个文件路径，例如: load data.json")
                continue
            files = parts[1:]
            total_added = 0
            for fp in files:
                if not os.path.isfile(fp):
                    print(f"[警告] 文件不存在：{fp}")
                    continue
                qa_list = load_qa_from_file(fp)
                if qa_list is not None:
                    added = qa_system.add_qa_list(qa_list, rebuild_index=True)
                    total_added += added
                    print(f"从 '{os.path.basename(fp)}' 加载了 {added} 个新的 QA 对。")
            print(f"共添加 {total_added} 个新的 QA 对。当前总数：{len(qa_system.qa_list)}")
        elif cmd0 == 'count':
            print(f"已加载 QA 对数：{len(qa_system.qa_list)}")
        elif cmd0 == 'ask':
            if not qa_system.qa_list:
                print("请先使用 load 命令或使用预置数据。")
                continue
            question = cmd[len('ask '):].strip()
            if question == "":
                print("问题为空。")
                continue
            answer, score, details = qa_system.ask(question)
            print(f"小助手：{answer}  (combined={score:.3f}, tfidf={details[1]:.3f}, lcs={details[2]:.3f})")
        elif cmd0 == 'history':
            for i, h in enumerate(qa_system.history, 1):
                print(f"{i}. Q: {h['question']} -> A: {h['answer']} (combined={h['score']:.3f}, tfidf={h['tfidf']:.3f}, lcs={h['lcs']:.3f})")
        elif cmd0 == 'save_history':
            if len(parts) < 2:
                print("请指定输出文件名，例如: save_history history.json")
                continue
            out = parts[1]
            try:
                with open(out, 'w', encoding='utf-8') as f:
                    json.dump(qa_system.history, f, ensure_ascii=False, indent=2)
                print(f"历史已保存到 {out}")
            except Exception as e:
                print(f"[错误] 保存失败：{e}")
        elif cmd0 == 'help':
            print_help()
        elif cmd0 == 'exit':
            print("退出。")
            break
        else:
            # 非命令则视为直接提问（若有数据）
            if qa_system.qa_list:
                question = cmd
                answer, score, details = qa_system.ask(question)
                print(f"小助手：{answer}  (combined={score:.3f}, tfidf={details[1]:.3f}, lcs={details[2]:.3f})")
            else:
                print("未知命令。输入 'help' 查看可用命令。")
if __name__ == '__main__':
    main()
