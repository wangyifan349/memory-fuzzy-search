# -*- coding: utf-8 -*-
import os
import json
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

########################################
# 1. LCS 检索
########################################

def compute_lcs_length(str_a: str, str_b: str) -> int:
    len_a = len(str_a)
    len_b = len(str_b)
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            if str_a[i - 1] == str_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len_a][len_b]

def retrieve_by_lcs(
    query_text: str,
    qa_pairs: List[Tuple[str, str]]
) -> Tuple[str, float]:
    best_score = -1.0
    best_answer = ""
    for db_question, db_answer in qa_pairs:
        lcs_len = compute_lcs_length(query_text, db_question)
        denom = max(len(query_text), len(db_question), 1)
        score = lcs_len / denom
        if score > best_score:
            best_score = score
            best_answer = db_answer
    return best_answer, best_score

########################################
# 2. TF + Cosine 检索
########################################

def build_tf_index(
    qa_pairs: List[Tuple[str, str]]
) -> Tuple[np.ndarray, dict, List[str], List[str]]:
    question_list = [q for q, _ in qa_pairs]
    answer_list   = [a for _, a in qa_pairs]
    # 构建词表
    vocabulary = {}
    for question in question_list:
        for token in question.split():
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    vocab_size = len(vocabulary)
    num_q = len(question_list)

    tf_matrix = np.zeros((num_q, vocab_size), dtype=np.float32)
    for idx, question in enumerate(question_list):
        tokens = question.split()
        for token in tokens:
            tf_matrix[idx, vocabulary[token]] += 1.0
        if tokens:
            tf_matrix[idx] /= len(tokens)

    return tf_matrix, vocabulary, question_list, answer_list

def retrieve_by_tf_cosine(
    query_text: str,
    tf_matrix: np.ndarray,
    vocabulary: dict,
    question_list: List[str],
    answer_list: List[str]
) -> Tuple[str, float]:
    query_vector = np.zeros(len(vocabulary), dtype=np.float32)
    tokens = query_text.split()
    for token in tokens:
        if token in vocabulary:
            query_vector[vocabulary[token]] += 1.0
    if tokens and query_vector.sum() > 0:
        query_vector /= len(tokens)

    tf_norms = np.linalg.norm(tf_matrix, axis=1)
    q_norm = np.linalg.norm(query_vector)
    denom = tf_norms * q_norm
    dots = tf_matrix.dot(query_vector)
    sims = np.where(denom > 0, dots / denom, 0.0)

    best_idx = int(np.argmax(sims))
    return answer_list[best_idx], float(sims[best_idx])

########################################
# 3. Sentence-BERT 检索（本地离线版）
########################################

def build_sbert_index(
    qa_pairs: List[Tuple[str, str]],
    local_model_dir: str = "./local_sbert/paraphrase-multilingual-MiniLM-L12-v2"
) -> Tuple[np.ndarray, SentenceTransformer, List[str], List[str]]:
    question_list = [q for q, _ in qa_pairs]
    answer_list   = [a for _, a in qa_pairs]
    model = SentenceTransformer(local_model_dir)
    embeddings = model.encode(
        question_list,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings, model, question_list, answer_list

def retrieve_by_sbert(
    query_text: str,
    embeddings: np.ndarray,
    model: SentenceTransformer,
    answer_list: List[str]
) -> Tuple[str, float]:
    q_emb = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]
    sims = embeddings.dot(q_emb)
    best_idx = int(np.argmax(sims))
    return answer_list[best_idx], float(sims[best_idx])

########################################
# 读取或初始化本地问答数据库
########################################

def load_local_qa_database(db_path: str = "qa_database.json"
) -> List[Tuple[str, str]]:
    if os.path.isfile(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 转成 List[Tuple[str,str]]
        return [(item["question"], item["answer"]) for item in data]
    else:
        return []

def save_local_qa_database(
    qa_pairs: List[Tuple[str, str]],
    db_path: str = "qa_database.json"
):
    data = [{"question": q, "answer": a} for q, a in qa_pairs]
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

########################################
# 主循环：持续问答
########################################

def main():
    # 1. 加载本地 DB（如果有），再加载默认示例
    local_db = load_local_qa_database()
    default_qa = [
        ("今天天气如何？", "今天晴好，适合出行。"),
        ("明天会下雨吗？", "明天有小雨，记得带伞。"),
        ("怎么学习 Python？", "建议看官方文档并做实战项目。")
    ]
    qa_pairs = local_db + default_qa

    # 2. 构建索引（LCS 不需要预构建）
    tf_matrix, vocabulary, questions_tf, answers_tf = build_tf_index(qa_pairs)
    embeddings_sbert, sbert_model, questions_sbert, answers_sbert = build_sbert_index(qa_pairs)

    print("请输入您的问题（输入 exit 或 quit 退出）：")
    while True:
        user_input = input(">> ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        # LCS 检索
        ans_lcs, score_lcs = retrieve_by_lcs(user_input, qa_pairs)
        # TF + Cosine 检索
        ans_tf, score_tf = retrieve_by_tf_cosine(
            user_input, tf_matrix, vocabulary, questions_tf, answers_tf
        )
        # S-BERT 检索
        ans_sbert, score_sbert = retrieve_by_sbert(
            user_input, embeddings_sbert, sbert_model, answers_sbert
        )

        print(f"[LCS]    答案 = {ans_lcs}，相似度 = {score_lcs:.3f}")
        print(f"[TF]     答案 = {ans_tf}，相似度 = {score_tf:.3f}")
        print(f"[S-BERT] 答案 = {ans_sbert}，相似度 = {score_sbert:.3f}")

    # 程序结束前，可选择保存当前 QA 对到本地
    # 如果你在运行过程中动态新增问题/答案，可以在此调用 save_local_qa_database
    # save_local_qa_database(qa_pairs)

if __name__ == "__main__":
    main()
