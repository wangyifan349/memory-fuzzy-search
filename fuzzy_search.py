import math
from collections import Counter
from typing import Dict, Tuple
import jieba  # 引入jieba库进行中文分词

# 示例QA字典
qa_dict = {
    "什么是 X25519？": "X25519 是一种基于 Curve25519 椭圆曲线的 Diffie–Hellman 公钥交换算法，具有高性能、小密钥尺寸、内置抗侧信道攻击设计以及恒时操作特性，支持快速安全的密钥协商，已被广泛采用于 TLS 1.3、SSH、Signal 协议以及各种加密库中。",
    "ChaCha20 加密算法有哪些特点？": "ChaCha20 是 Google 设计的 256 位密钥流密码，具有极高的安全性（至今无实用性攻击）、出色的软硬件性能（在没有 AES 硬件加速的设备上也能高效运行）、简单扁平的算法结构（便于验证及避免实现错误和侧信道泄露），常与 Poly1305 组合为 AEAD 模式用于 TLS、VPN、SSH 等场景。",
    "神经细胞（神经元）的基本结构包括哪些部分？": "神经元由细胞体（含细胞核和大部分胞器，负责基因表达和能量代谢）、树突（接收来自其他神经元或感受器的突触输入并将信号传导至胞体）、轴突（从胞体延伸，负责将动作电位快速传导至远端目标）和突触（轴突末端处的信号传递结构，通过释放神经递质在化学或电学形式下完成与下一个细胞的通讯）四大部分构成。",
    "如何防止 SQL 注入攻击？": "防范 SQL 注入的最佳实践包括始终使用参数化查询（Prepared Statements）或 ORM 提供的安全接口以避免动态拼接 SQL；对所有用户输入进行严格的类型、格式和长度校验；在数据库层面采用最小权限原则，仅授予应用所需的最小 CRUD 权限；另外可以部署 Web 应用防火墙（WAF）和数据库审计工具，对异常或可疑查询进行实时监控和阻断。"
}


# 简易词频向量化函数，将文本转为小写分词后的Counter对象
def text_to_vector(text: str) -> Counter:
    # 使用jieba进行中文分词
    words = list(jieba.cut(text))
    # 返回词频统计
    return Counter(words)

# 计算余弦相似度函数，输入两个词频Counter向量
def cosine_similarity(vec1: Counter, vec2: Counter) -> float:
    # 计算两个向量的点积
    intersection = set(vec1) & set(vec2)
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    # 计算两个向量的欧几里得范数
    sum1 = sum([v**2 for v in vec1.values()])
    sum2 = sum([v**2 for v in vec2.values()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    # 避免除零
    if not denominator:
        return 0.0
    else:
        return numerator / denominator

# 计算编辑距离（Levenshtein距离）
def levenshtein_distance(s1: str, s2: str) -> int:
    s1 = s1.lower()
    s2 = s2.lower()
    m, n = len(s1), len(s2)
    # 创建动态规划矩阵
    dp = [[0] * (n+1) for _ in range(m+1)]
    # 初始化第一列（空串s2）
    for i in range(m+1):
        dp[i][0] = i
    # 初始化第一行（空串s1）
    for j in range(n+1):
        dp[0][j] = j
    # 填充DP矩阵
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1]+1, dp[i][j-1]+1, dp[i-1][j]+1)
    return dp[m][n]

# 计算最长公共子序列长度（LCS）
def lcs_length(s1: str, s2: str) -> int:
    s1 = s1.lower()
    s2 = s2.lower()
    m, n = len(s1), len(s2)
    # 创建DP矩阵
    dp = [[0]*(n+1) for _ in range(m+1)]
    # 计算LCS
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

# 组合相似度计算函数，结合LCS和编辑距离，返回0到1的相似度
def combined_similarity(s1: str, s2: str, weight_lcs=0.5, weight_lev=0.5) -> float:
    if not s1 and not s2:
        return 1.0  # 两空字符串相似度1
    # 计算编辑距离
    lev_dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    # 归一化编辑距离转相似度
    lev_sim = 1 - lev_dist / max_len if max_len > 0 else 1.0
    # 计算LCS相似度
    lcs_len = lcs_length(s1, s2)
    lcs_sim = lcs_len / max_len if max_len > 0 else 1.0
    # 返回加权平均相似度
    return weight_lcs * lcs_sim + weight_lev * lev_sim

# 主搜索函数，输入查询，qa字典，匹配方法，返回匹配答案和分数
def search_qa(query: str, qa_data: Dict[str, str], method: str = 'vector') -> Tuple[str, float]:
    max_score = -1
    best_answer = "抱歉，未找到相关答案。"
    query = query.strip()
    if method == 'vector':  # 词频向量+余弦相似度
        query_vec = text_to_vector(query)
        for q, a in qa_data.items():
            q_vec = text_to_vector(q)
            score = cosine_similarity(query_vec, q_vec)
            if score > max_score:
                max_score = score
                best_answer = a
    elif method == 'lcs_lev':  # LCS+编辑距离组合相似度
        for q, a in qa_data.items():
            score = combined_similarity(query, q)
            if score > max_score:
                max_score = score
                best_answer = a
    else:
        raise ValueError("method参数必须是'vector'或'lcs_lev'")
    return best_answer, max_score

if __name__ == "__main__":
    print("欢迎使用QA问答系统，请输入您的问题（输入exit退出）：")
    while True:
        user_query = input("请输入问题：").strip()
        if user_query.lower() == "exit":
            print("感谢使用，再见！")
            break
        if not user_query:
            print("请输入有效的问题！")
            continue

        # 选择匹配方式: 这里示例先用两种都展示
        answer_vec, score_vec = search_qa(user_query, qa_dict, method='vector')
        print(f"[词向量+余弦] 匹配度: {score_vec:.3f}，答案:\n{answer_vec}\n")

        answer_lcs, score_lcs = search_qa(user_query, qa_dict, method='lcs_lev')
        print(f"[LCS+编辑距离] 匹配度: {score_lcs:.3f}，答案:\n{answer_lcs}\n")
