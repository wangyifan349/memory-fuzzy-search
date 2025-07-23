"""
示例脚本：使用 NLTK 分词 + 自定义 n-gram Tokenizer，结合 sklearn 的 TF-IDF 和余弦相似度，
来区分句子中词序不同但词集合相同的情况（例如“我爱你” vs “你爱我”）。

主要功能：
1. 用 NLTK 做基本分词（word tokenization）并过滤非字母或数字字符。
2. 基于分词结果生成任意范围的 n-gram（unigram、bigram、…）。
3. 将这些 n-gram Token 当作特征，交给 TfidfVectorizer 计算 TF-IDF 向量。
4. 用余弦相似度比较句子向量，能区分词序差异。
运行依赖：
- nltk
- scikit-learn
"""
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# ----------------------------------------------------------------
# 1. 下载 NLTK 需要的资源，只需执行一次即可
#    punkt：用于 word_tokenize 分词
# ----------------------------------------------------------------
nltk.download('punkt', quiet=True)
# ----------------------------------------------------------------
# 2. 定义自定义的 Tokenizer：
#    输入：一句文本（string）和 ngram_range（tuple，min_n, max_n）
#    输出：一个由 unigram/bigram/.../n-gram 组成的 token 列表
# ----------------------------------------------------------------
def nltk_tokenizer_with_ngrams(text, ngram_range=(1, 2)):
    """
    用 NLTK word_tokenize 把 text 切分成基本词 token，
    然后生成 min_n 到 max_n 各阶 n-gram（以 '_' 连接子词）并返回所有 tokens。

    参数：
      text (str): 待处理的原始句子
      ngram_range (tuple(int, int)): n-gram 的最小 n 和最大 n，例如 (1,2) 表示 unigram + bigram

    返回：
      List[str]: 包含所有 unigram、bigram、…、n-gram 的 token 列表
    """
    # 1）分词并变小写，过滤掉标点或其它非字母数字字符
    raw_tokens = nltk.word_tokenize(text)
    tokens = [t.lower() for t in raw_tokens if t.isalnum()]

    min_n, max_n = ngram_range
    output_tokens = []

    # 2）生成各阶 n-gram
    for n in range(min_n, max_n + 1):
        if n == 1:
            # unigram：直接把 tokens 加入输出
            output_tokens.extend(tokens)
        else:
            # n-gram：滑动窗口
            for i in range(len(tokens) - n + 1):
                gram = tokens[i : i + n]
                # 用下划线连接为一个 token
                output_tokens.append("_".join(gram))

    return output_tokens

# ----------------------------------------------------------------
# 3. 构建 TF-IDF Vectorizer：
#    - 使用自定义 tokenizer
#    - 关闭内部 lowercase/token_pattern，让我们全权控制
# ----------------------------------------------------------------
vectorizer = TfidfVectorizer(
    tokenizer=lambda txt: nltk_tokenizer_with_ngrams(txt, ngram_range=(1, 2)),
    lowercase=False,    # 我们在 tokenizer 已经小写过了
    token_pattern=None  # 禁用默认正则，使用自定义 tokenizer
)

# ----------------------------------------------------------------
# 4. 示例句子列表
# ----------------------------------------------------------------
sentences = [
    "我 爱 你",
    "你 爱 我",
    "今天 天气 很 好",          # 额外示例
    "天气 今天 很 好",          # 同样内容不同顺序
    "NLTK 并 不 支持 中文 分词", # 测试中英混合
    "中文 分词 NLTK 并 不 支持"
]

# ----------------------------------------------------------------
# 5. 训练 TF-IDF 模型：fit_transform 会返回 (n_sentences x n_features) 的稀疏矩阵
# ----------------------------------------------------------------
tfidf_matrix = vectorizer.fit_transform(sentences)

# ----------------------------------------------------------------
# 6. 打印所有特征名（unigram + bigram）
# ----------------------------------------------------------------
print("所有特征名（unigram + bigram）：")
print(vectorizer.get_feature_names_out())
print()

# ----------------------------------------------------------------
# 7. 定义一个辅助函数：计算并展示任意两句的余弦相似度
# ----------------------------------------------------------------
def show_similarity(idx1, idx2):
    """
    计算 sentences[idx1] 与 sentences[idx2] 的余弦相似度并打印结果。
    """
    vec1 = tfidf_matrix[idx1:idx1+1]
    vec2 = tfidf_matrix[idx2:idx2+1]
    sim = cosine_similarity(vec1, vec2)[0, 0]
    print(f"“{sentences[idx1]}” vs “{sentences[idx2]}” 的余弦相似度 = {sim:.4f}")

# ----------------------------------------------------------------
# 8. 演示对比：只用 unigram 时 & 用 bigram 时的差异
#    直接调用 show_similarity，比较不同顺序句子的相似度
# ----------------------------------------------------------------
print("=== 对比相同词但顺序不同的句子余弦相似度 ===")
show_similarity(0, 1)  # 我 爱 你 vs 你 爱 我
show_similarity(2, 3)  # 今天 天气 很 好 vs 天气 今天 很 好
show_similarity(4, 5)  # NLTK 并 不 支持 中文 分词 vs 中文 分词 NLTK 并 不 支持

# ----------------------------------------------------------------
# 9. 附加示例：比较“我 爱 你”与“我 爱 你”自身（应该完全相似）
# ----------------------------------------------------------------
print("\n=== 自身对比（应该是 1.0000） ===")
show_similarity(0, 0)
