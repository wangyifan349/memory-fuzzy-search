"""
pip install jieba numpy python-Levenshtein textdistance
"""
import jieba
import numpy as np
import Levenshtein  # 这个库可以继续用来计算编辑距离，不用也行，我也加了纯Python版

# QA 数据示例（英文）
qa_dict = {
    "What is X25519?": "X25519 is a Diffie–Hellman public key exchange algorithm based on the Curve25519 elliptic curve...",
    "What are the characteristics of the ChaCha20 encryption algorithm?": "ChaCha20 is a 256-bit stream cipher designed by Google...",
    "What are the basic structures of neurons?": "Neurons consist of four main parts: the cell body, dendrites, axon, and synapses.",
    "How to prevent SQL injection attacks?": "Best practices include using parameterized queries, strict input validation, least privilege principle..."
}

# -------------- jieba分词 --------------
def tokenize(text):
    return list(jieba.cut(text))


# -------------- 词袋向量 + 余弦相似度 --------------
def build_vocab(texts):
    vocab = set()
    for t in texts:
        vocab.update(tokenize(t))
    return sorted(vocab)

def text_to_tf_vector(text, vocab):
    words = tokenize(text)
    word_count = {w: 0 for w in vocab}
    for w in words:
        if w in word_count:
            word_count[w] += 1
    vec = np.array([word_count[w] for w in vocab], dtype=float)
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec

def cosine_similarity_np(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def tf_vector_match(query, qa_data):
    questions = list(qa_data.keys())
    vocab = build_vocab(questions + [query])
    query_vec = text_to_tf_vector(query, vocab)

    max_score = -1
    best_answer = "Sorry, no relevant answer found."

    for q in questions:
        q_vec = text_to_tf_vector(q, vocab)
        score = cosine_similarity_np(query_vec, q_vec)
        if score > max_score:
            max_score = score
            best_answer = qa_data[q]
    return best_answer, max_score

# -------------- 纯Python编辑距离（Levenshtein距离）实现 --------------
def levenshtein_distance_py(s1, s2):
    s1 = s1.lower()
    s2 = s2.lower()
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1,m+1):
        for j in range(1,n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) +1
    return dp[m][n]

# -------------- 纯Python最长公共子序列长度实现 --------------
def lcs_length_py(s1, s2):
    s1 = s1.lower()
    s2 = s2.lower()
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]+1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# -------------- 编辑距离 + LCS加权相似度 --------------
def combined_sim_py(s1, s2, w_lcs=0.5, w_lev=0.5):
    if not s1 and not s2:
        return 1.0
    lev_dist = levenshtein_distance_py(s1, s2)
    max_len = max(len(s1), len(s2))
    lev_sim = 1 - lev_dist/max_len if max_len>0 else 1.0
    lcs_sim = lcs_length_py(s1, s2)/max_len if max_len>0 else 1.0
    return w_lcs*lcs_sim + w_lev*lev_sim

def lcs_lev_match_py(query, qa_data):
    max_score = -1
    best_answer = "Sorry, no relevant answer found."
    for q in qa_data:
        score = combined_sim_py(query, q)
        if score > max_score:
            max_score = score
            best_answer = qa_data[q]
    return best_answer, max_score


# ---------------- 主程序 -----------------
if __name__ == "__main__":
    print("The QA system supports two matching methods:")
    print("1 - TF-Bag of Words Vector + Cosine Similarity")
    print("2 - Edit Distance + Longest Common Subsequence Weighted (Pure Python implementation)")
    print("Type 'exit' to exit the system")

    while True:
        query = input("\nPlease enter your question: ").strip()
        if query.lower() == "exit":
            print("Thank you for using, goodbye!")
            break
        if not query:
            print("Please enter a valid question!")
            continue

        method = input("Please choose a matching method (1 or 2): ").strip()
        if method == "1":
            answer, score = tf_vector_match(query, qa_dict)
            print(f"[TF-Bag of Words + Cosine] Similarity: {score:.4f}\nAnswer: {answer}")
        elif method == "2":
            answer, score = lcs_lev_match_py(query, qa_dict)
            print(f"[Edit Distance + LCS Weighted] Similarity: {score:.4f}\nAnswer: {answer}")
        else:
            print("Invalid choice, please enter 1 or 2.")
