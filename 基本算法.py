import math
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

# 1. 余弦相似度（Cosine Similarity）
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)                   # 向量点积
    norm1 = np.linalg.norm(vec1)                       # vec1的L2范数
    norm2 = np.linalg.norm(vec2)                       # vec2的L2范数
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

# 2. 点积（Dot Product）
def dot_product(vec1, vec2):
    return np.dot(vec1, vec2)                          # 计算两个向量的点积

# 3. 欧式距离（Euclidean Distance）
def euclidean_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    diff = vec1 - vec2
    return np.linalg.norm(diff)                        # 计算两个向量的欧式距离

# 4. 曼哈顿距离（Manhattan Distance）
def manhattan_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    diff = vec1 - vec2
    return np.sum(np.abs(diff))                        # 计算两个向量的曼哈顿距离

# 5. 马氏距离（Mahalanobis Distance）
def mahalanobis_distance(vec1, vec2, covariance_matrix):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    cov_inv = np.linalg.inv(covariance_matrix)
    return mahalanobis(vec1, vec2, cov_inv)            # 计算马氏距离

# 6. 杰卡德相似度（Jaccard Similarity）
def jaccard_similarity(set1, set2):
    set1 = set(set1)
    set2 = set(set2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 1.0
    return len(intersection) / len(union)              # 计算杰卡德相似度

# 7. 汉明距离（Hamming Distance）
def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("字符串长度必须相等")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2)) # 计算汉明距离

# 8. 向量归一化（Normalization）
def normalize_vector(vec):
    vec = np.array(vec)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm                                  # 归一化向量

# 9. 最长公共子序列（Longest Common Subsequence, LCS）
def longest_common_subsequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]               # 初始化DP表
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]                                    # 返回LCS长度

# 10. 编辑距离（Levenshtein Distance）
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]               # 初始化DP表
    for i in range(m+1):
        dp[i][0] = i                                   
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,                        # 删除
                dp[i][j-1] + 1,                        # 插入
                dp[i-1][j-1] + cost                    # 替换或匹配
            )
    return dp[m][n]                                    # 返回编辑距离

# 11. 斐波那契数列 - 递归版
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)  # 递归计算

# 12. 斐波那契数列 - 动态规划版
def fibonacci_dp(n):
    if n <= 1:
        return n
    fib = [0]*(n+1)
    fib[1] = 1
    for i in range(2, n+1):
        fib[i] = fib[i-1] + fib[i-2]
    return fib[n]                                       # 返回斐波那契第n项

# 13. KMP算法前缀表计算
def kmp_compute_prefix(pattern):
    length = 0
    lps = [0]*len(pattern)
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length-1]
            else:
                lps[i] = 0
                i += 1
    return lps                                           # 返回部分匹配表

# 14. KMP字符串匹配
def kmp_search(text, pattern):
    if not pattern:
        return []
    lps = kmp_compute_prefix(pattern)
    result = []
    i = j = 0
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == len(pattern):
                result.append(i - j)
                j = lps[j-1]
        else:
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
    return result                                        # 返回匹配位置列表

# 15. 主成分分析（PCA）
def pca_reduce(data, n_components):
    pca_model = PCA(n_components=n_components)
    reduced = pca_model.fit_transform(data)
    return reduced                                       # 返回降维后数据

# 16. t-SNE降维
def tsne_reduce(data, n_components=2, perplexity=30, random_state=42):
    tsne_model = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    reduced = tsne_model.fit_transform(data)
    return reduced                                       # 返回降维后数据

# 17. UMAP降维
def umap_reduce(data, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    reduced = reducer.fit_transform(data)
    return reduced                                       # 返回降维后数据

# 测试示例
if __name__ == "__main__":
    vec1 = [1, 2, 3]
    vec2 = [4, 5, 6]
    set1 = [1, 2, 3]
    set2 = [2, 3, 4]
    s1 = "kitten"
    s2 = "sitting"
    str1 = "karolin"
    str2 = "kathrin"
    text = "ababcabcabababd"
    pattern = "ababd"
    data_samples = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4]])

    print("余弦相似度:", cosine_similarity(vec1, vec2))
    print("点积:", dot_product(vec1, vec2))
    print("欧式距离:", euclidean_distance(vec1, vec2))
    print("曼哈顿距离:", manhattan_distance(vec1, vec2))
    print("杰卡德相似度:", jaccard_similarity(set1, set2))
    print("汉明距离:", hamming_distance(str1, str2))

    data_for_cov = np.array([vec1, vec2])
    cov_matrix = np.cov(data_for_cov, rowvar=False)
    print("马氏距离:", mahalanobis_distance(vec1, vec2, cov_matrix))
    print("归一化向量:", normalize_vector(vec1))

    print("最长公共子序列长度:", longest_common_subsequence("abcde", "ace"))
    print("编辑距离:", edit_distance(s1, s2))
    print("斐波那契数列第10项(递归):", fibonacci_recursive(10))
    print("斐波那契数列第10项(DP):", fibonacci_dp(10))
    print("KMP匹配结果:", kmp_search(text, pattern))

    print("PCA降维到2维:\n", pca_reduce(data_samples, n_components=2))
    print("t-SNE降维到2维:\n", tsne_reduce(data_samples))
    print("UMAP降维到2维:\n", umap_reduce(data_samples))



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# 示例文档列表
documents = [
    "I love machine learning and NLP",
    "Word2Vec captures semantic relationships",
    "Text processing is fun"
]
# 1. TF-IDF（词频-逆文档频率）
def compute_tfidf(docs):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names
# 2. 词袋模型（Bag of Words）
def compute_bag_of_words(docs):
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    return count_matrix, feature_names
# 3. Word2Vec 训练
def train_word2vec(sentences, vector_size=100, window=5, min_count=1):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, sg=0)
    return model
# 计算TF-IDF
tfidf_matrix, tfidf_features = compute_tfidf(documents)
print("TF-IDF 特征名称:\n", tfidf_features)
print("TF-IDF 矩阵:\n", tfidf_matrix.toarray())
# 计算词袋模型
bow_matrix, bow_features = compute_bag_of_words(documents)
print("\n词袋模型 特征名称:\n", bow_features)
print("词袋模型 矩阵:\n", bow_matrix.toarray())
# 处理文本用于Word2Vec训练
processed_sentences = [simple_preprocess(doc) for doc in documents]
print("\n处理后的句子（分词结果）:")
for idx, sentence in enumerate(processed_sentences):
    print(f"句子 {idx + 1}: {sentence}")
# 训练Word2Vec模型
w2v_model = train_word2vec(processed_sentences)

# 展示Word2Vec示例
word = 'machine'
if word in w2v_model.wv:
    print(f"\nWord2Vec: '{word}' 的向量表示:\n", w2v_model.wv[word])
else:
    print(f"\nWord2Vec: '{word}' 不在模型词典中。")

# 展示与指定词最相似词汇
similar_words = w2v_model.wv.most_similar(word, topn=5) if word in w2v_model.wv else []
print(f"\n与 '{word}' 最相似的词语:")
for similar_word, similarity in similar_words:
    print(f"词语: {similar_word}, 相似度: {similarity}")

# 打印词与词之间的相似度
word_pairs = [('machine', 'learning'), ('text', 'processing'), ('love', 'fun')]
print("\n词与词之间的相似度:")
for w1, w2 in word_pairs:
    if w1 in w2v_model.wv and w2 in w2v_model.wv:
        similarity = w2v_model.wv.similarity(w1, w2)
        print(f"'{w1}' 与 '{w2}' 的相似度: {similarity}")
    else:
        print(f"'{w1}' 或 '{w2}' 不在模型词典中。")

