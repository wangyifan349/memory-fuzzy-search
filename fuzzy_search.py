"""
pip install jieba numpy python-Levenshtein textdistance
"""

import jieba
import numpy as np
import Levenshtein
import textdistance

# Sample QA Dictionary
qa_dict = {
    "What is X25519?": "X25519 is a Diffie–Hellman public key exchange algorithm based on the Curve25519 elliptic curve, known for its high performance, small key size, built-in resistance to side-channel attacks, and constant-time operation. It supports fast and secure key agreement and is widely used in TLS 1.3, SSH, Signal protocol, and various cryptographic libraries.",
    "What are the characteristics of the ChaCha20 encryption algorithm?": "ChaCha20 is a 256-bit stream cipher designed by Google, offering high security (no practical attacks to date), excellent performance on both software and hardware (efficiently operates on devices without AES hardware acceleration), and a simple flat algorithm structure (facilitating verification and avoiding implementation errors and side-channel leaks), often combined with Poly1305 to form AEAD modes for TLS, VPN, SSH, and other scenarios.",
    "What are the basic structures of neurons?": "Neurons consist of four main parts: the cell body (containing the nucleus and most organelles, responsible for gene expression and energy metabolism), dendrites (receiving synaptic inputs from other neurons or receptors and transmitting signals to the cell body), axon (extending from the cell body, responsible for rapidly transmitting action potentials to distant targets), and synapses (signal transmission structures at the axon terminals that communicate with the next cell through the release of neurotransmitters in chemical or electrical forms).",
    "How to prevent SQL injection attacks?": "Best practices for preventing SQL injection include always using parameterized queries (Prepared Statements) or secure interfaces provided by ORM to avoid dynamically constructing SQL; strictly validating all user inputs in terms of type, format, and length; applying the principle of least privilege at the database level by granting only the minimum necessary CRUD permissions; additionally, deploying web application firewalls (WAF) and database auditing tools to monitor and block abnormal or suspicious queries in real time."
}

# ---------------------------- Bag of Words Vector + Cosine Similarity ----------------------------

def tokenize(text):
    return list(jieba.cut(text))

def build_vocab(texts):
    vocab = set()
    for t in texts:
        vocab.update(tokenize(t))
    return sorted(vocab)

def text_to_tf_vector(text, vocab):
    words = tokenize(text)
    word_count = {w:0 for w in vocab}
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


# ---------------------------- Edit Distance + Longest Common Subsequence Weighted ----------------------------

def levenshtein_distance(s1, s2):
    return Levenshtein.distance(s1.lower(), s2.lower())

def lcs_length(s1, s2):
    return textdistance.lcsseq.len(s1.lower(), s2.lower())

def combined_sim(s1, s2, w_lcs=0.5, w_lev=0.5):
    if not s1 and not s2:
        return 1.0
    lev_dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    lev_sim = 1 - lev_dist/max_len if max_len>0 else 1.0
    lcs_sim = lcs_length(s1, s2)/max_len if max_len>0 else 1.0
    return w_lcs*lcs_sim + w_lev*lev_sim

def lcs_lev_match(query, qa_data):
    max_score = -1
    best_answer = "Sorry, no relevant answer found."
    for q in qa_data:
        score = combined_sim(query, q)
        if score > max_score:
            max_score = score
            best_answer = qa_data[q]
    return best_answer, max_score

# ---------------------------- Main Program ----------------------------

if __name__ == "__main__":
    print("The QA system supports two matching methods:")
    print("1 - TF-Bag of Words Vector + Cosine Similarity")
    print("2 - Edit Distance + Longest Common Subsequence Weighted")
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
            answer, score = lcs_lev_match(query, qa_dict)
            print(f"[Edit Distance + LCS Weighted] Similarity: {score:.4f}\nAnswer: {answer}")
        else:
            print("Invalid choice, please enter 1 or 2.")
