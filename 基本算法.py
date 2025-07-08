import math
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# -----------------------------------------------
# Similarity and Distance Measures
# -----------------------------------------------

def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.
    """
    vec1 = np.array(vec1)  # Convert list to NumPy array
    vec2 = np.array(vec2)  # Convert list to NumPy array
    dot_product = np.dot(vec1, vec2)  # Compute dot product
    norm1 = np.linalg.norm(vec1)  # Compute the L2 norm of vec1
    norm2 = np.linalg.norm(vec2)  # Compute the L2 norm of vec2
    if norm1 == 0 or norm2 == 0:
        return 0  # Return 0 if either vector is zero
    return dot_product / (norm1 * norm2)  # Return cosine similarity

def dot_product(vec1, vec2):
    """
    Calculate the dot product between two vectors.
    """
    vec1 = np.array(vec1)  # Convert list to NumPy array
    vec2 = np.array(vec2)  # Convert list to NumPy array
    product = np.dot(vec1, vec2)  # Compute dot product
    return product  # Return dot product

def euclidean_distance(vec1, vec2):
    """
    Calculate the Euclidean distance between two vectors.
    """
    vec1 = np.array(vec1)  # Convert list to NumPy array
    vec2 = np.array(vec2)  # Convert list to NumPy array
    diff = vec1 - vec2  # Compute difference between vectors
    distance = np.linalg.norm(diff)  # Compute L2 norm of difference
    return distance  # Return Euclidean distance

def manhattan_distance(vec1, vec2):
    """
    Calculate the Manhattan distance between two vectors.
    """
    vec1 = np.array(vec1)  # Convert list to NumPy array
    vec2 = np.array(vec2)  # Convert list to NumPy array
    diff = vec1 - vec2  # Compute difference between vectors
    distance = np.sum(np.abs(diff))  # Compute sum of absolute differences
    return distance  # Return Manhattan distance

def mahalanobis_distance(vec1, vec2, covariance_matrix):
    """
    Calculate the Mahalanobis distance between two vectors given a covariance matrix.
    """
    vec1 = np.array(vec1)  # Convert list to NumPy array
    vec2 = np.array(vec2)  # Convert list to NumPy array
    cov_inv = np.linalg.inv(covariance_matrix)  # Invert the covariance matrix
    distance = mahalanobis(vec1, vec2, cov_inv)  # Compute Mahalanobis distance
    return distance  # Return Mahalanobis distance

def jaccard_similarity(set1, set2):
    """
    Calculate the Jaccard similarity between two sets.
    """
    set1 = set(set1)  # Convert list to set
    set2 = set(set2)  # Convert list to set
    intersection = set1.intersection(set2)  # Find intersection
    union = set1.union(set2)  # Find union
    if not union:
        return 1.0  # If both sets are empty, return 1.0
    similarity = float(len(intersection)) / len(union)  # Compute Jaccard similarity
    return similarity  # Return Jaccard similarity

def hamming_distance(str1, str2):
    """
    Calculate the Hamming distance between two strings of equal length.
    """
    if len(str1) != len(str2):
        raise ValueError("Strings must be of equal length")  # Raise error if lengths differ
    distance = 0  # Initialize distance
    for c1, c2 in zip(str1, str2):
        if c1 != c2:
            distance += 1  # Increment distance for each differing character
    return distance  # Return Hamming distance

def normalize_vector(vec):
    """
    Normalize a vector to have a unit norm.
    """
    vec = np.array(vec)  # Convert list to NumPy array
    norm = np.linalg.norm(vec)  # Compute L2 norm of vector
    if norm == 0:
        return vec  # If norm is zero, return the original vector
    normalized_vec = vec / norm  # Divide vector by its norm
    return normalized_vec  # Return normalized vector

# -----------------------------------------------
# String Algorithms
# -----------------------------------------------

def longest_common_subsequence(s1, s2):
    """
    Find the length of the longest common subsequence between two strings.
    """
    m = len(s1)  # Length of first string
    n = len(s2)  # Length of second string
    dp = []  # Initialize DP table
    for i in range(m + 1):
        dp_row = []
        for j in range(n + 1):
            dp_row.append(0)
        dp.append(dp_row)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1  # Characters match
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  # Take maximum of left and top cell
    return dp[m][n]  # Return length of LCS

def edit_distance(s1, s2):
    """
    Calculate the Levenshtein edit distance between two strings.
    """
    m = len(s1)  # Length of first string
    n = len(s2)  # Length of second string
    dp = []  # Initialize DP table
    for i in range(m + 1):
        dp_row = []
        for j in range(n + 1):
            dp_row.append(0)
        dp.append(dp_row)
    for i in range(m + 1):
        dp[i][0] = i  # Cost of deletions
    for j in range(n + 1):
        dp[0][j] = j  # Cost of insertions
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1  # Cost is 0 if characters match
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # Deletion
                dp[i][j - 1] + 1,        # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )
    return dp[m][n]  # Return edit distance

# -----------------------------------------------
# Sequence Algorithms
# -----------------------------------------------

def fibonacci_recursive(n):
    """
    Calculate the nth Fibonacci number using recursion.
    """
    if n <= 1:
        return n  # Base case
    result = fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)  # Recursive calls
    return result  # Return nth Fibonacci number

def fibonacci_dp(n):
    """
    Calculate the nth Fibonacci number using dynamic programming.
    """
    if n <= 1:
        return n  # Base case
    fib = []
    for i in range(n + 1):
        fib.append(0)
    fib[1] = 1  # Seed value
    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]  # Sum of previous two numbers
    return fib[n]  # Return nth Fibonacci number

# -----------------------------------------------
# Pattern Matching Algorithms
# -----------------------------------------------

def kmp_compute_prefix(pattern):
    """
    Compute the Longest Proper Prefix array for KMP algorithm.
    """
    length = 0  # Length of the previous longest prefix suffix
    lps = []  # Initialize LPS array
    for i in range(len(pattern)):
        lps.append(0)
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]  # Fallback in the LPS array
            else:
                lps[i] = 0
                i += 1
    return lps  # Return LPS array

def kmp_search(text, pattern):
    """
    Perform KMP search of a pattern within a text.
    """
    if not pattern:
        return []  # Return empty list if pattern is empty
    lps = kmp_compute_prefix(pattern)  # Compute LPS array
    result = []  # Initialize result list
    i = 0  # Index for text
    j = 0  # Index for pattern
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
        if j == len(pattern):
            result.append(i - j)  # Pattern found
            j = lps[j - 1]  # Fallback in the pattern
        elif i < len(text) and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]  # Fallback in the pattern
            else:
                i += 1  # Move to next character in text
    return result  # Return list of positions where pattern is found

# -----------------------------------------------
# Dimensionality Reduction Techniques
# -----------------------------------------------

def pca_reduce(data, n_components):
    """
    Reduce dimensionality of data using PCA.
    """
    pca_model = PCA(n_components=n_components)  # Initialize PCA model
    reduced_data = pca_model.fit_transform(data)  # Fit and transform data
    return reduced_data  # Return reduced data

def tsne_reduce(data, n_components=2, perplexity=30, random_state=42):
    """
    Reduce dimensionality of data using t-SNE.
    """
    tsne_model = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)  # Initialize t-SNE model
    reduced_data = tsne_model.fit_transform(data)  # Fit and transform data
    return reduced_data  # Return reduced data

def umap_reduce(data, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Reduce dimensionality of data using UMAP.
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )  # Initialize UMAP reducer
    reduced_data = reducer.fit_transform(data)  # Fit and transform data
    return reduced_data  # Return reduced data

# -----------------------------------------------
# Text Vectorization and Word Embeddings
# -----------------------------------------------

def compute_tfidf(docs):
    """
    Compute TF-IDF matrix for a list of documents.
    """
    vectorizer = TfidfVectorizer()  # Initialize TF-IDF vectorizer
    tfidf_matrix = vectorizer.fit_transform(docs)  # Fit and transform documents
    feature_names = vectorizer.get_feature_names_out()  # Get feature names
    return tfidf_matrix, feature_names  # Return TF-IDF matrix and feature names

def compute_bag_of_words(docs):
    """
    Compute Bag of Words matrix for a list of documents.
    """
    vectorizer = CountVectorizer()  # Initialize CountVectorizer
    count_matrix = vectorizer.fit_transform(docs)  # Fit and transform documents
    feature_names = vectorizer.get_feature_names_out()  # Get feature names
    return count_matrix, feature_names  # Return BoW matrix and feature names

def train_word2vec(sentences, vector_size=100, window=5, min_count=1):
    """
    Train a Word2Vec model on a list of tokenized sentences.
    """
    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=0  # Use CBOW architecture
    )
    return model  # Return trained Word2Vec model

# -----------------------------------------------
# Main Function
# -----------------------------------------------

def main():
    """
    Main function to demonstrate all algorithms.
    """
    # Vectors and sets for similarity and distance measures
    vec1 = [1, 2, 3]
    vec2 = [4, 5, 6]
    set1 = [1, 2, 3]
    set2 = [2, 3, 4]
    str1 = "karolin"
    str2 = "kathrin"

    # Strings for string algorithms
    s1 = "kitten"
    s2 = "sitting"
    text = "ababcabcabababd"
    pattern = "ababd"

    # Data for dimensionality reduction
    data_samples = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [2, 3, 4]
    ])

    # Documents for text vectorization and embeddings
    documents = [
        "I love machine learning and NLP",
        "Word2Vec captures semantic relationships",
        "Text processing is fun"
    ]

    # Similarity and Distance Measures
    print("Cosine Similarity:", cosine_similarity(vec1, vec2))
    print("Dot Product:", dot_product(vec1, vec2))
    print("Euclidean Distance:", euclidean_distance(vec1, vec2))
    print("Manhattan Distance:", manhattan_distance(vec1, vec2))
    print("Jaccard Similarity:", jaccard_similarity(set1, set2))
    print("Hamming Distance:", hamming_distance(str1, str2))

    data_for_cov = np.array([vec1, vec2])
    cov_matrix = np.cov(data_for_cov, rowvar=False)
    print("Mahalanobis Distance:", mahalanobis_distance(vec1, vec2, cov_matrix))
    print("Normalized Vector:", normalize_vector(vec1))

    # String Algorithms
    print("Longest Common Subsequence Length:", longest_common_subsequence("abcde", "ace"))
    print("Edit Distance:", edit_distance(s1, s2))

    # Fibonacci Sequence
    n = 10
    print("Fibonacci Recursive (n=10):", fibonacci_recursive(n))
    print("Fibonacci DP (n=10):", fibonacci_dp(n))

    # KMP Algorithm
    print("KMP Search Result:", kmp_search(text, pattern))

    # Dimensionality Reduction
    pca_result = pca_reduce(data_samples, n_components=2)
    print("PCA Reduced Data:\n", pca_result)

    tsne_result = tsne_reduce(data_samples)
    print("t-SNE Reduced Data:\n", tsne_result)

    umap_result = umap_reduce(data_samples)
    print("UMAP Reduced Data:\n", umap_result)

    # Text Vectorization
    tfidf_matrix, tfidf_features = compute_tfidf(documents)
    print("TF-IDF Feature Names:\n", tfidf_features)
    print("TF-IDF Matrix:\n", tfidf_matrix.toarray())

    bow_matrix, bow_features = compute_bag_of_words(documents)
    print("\nBag of Words Feature Names:\n", bow_features)
    print("Bag of Words Matrix:\n", bow_matrix.toarray())

    # Word2Vec Training
    processed_sentences = []
    for doc in documents:
        tokens = simple_preprocess(doc)
        processed_sentences.append(tokens)
    print("\nProcessed Sentences:")
    for idx, sentence in enumerate(processed_sentences):
        print("Sentence {}: {}".format(idx + 1, sentence))

    w2v_model = train_word2vec(processed_sentences)
    word = 'machine'

    if word in w2v_model.wv:
        print("\nWord2Vec Vector for '{}':\n{}".format(word, w2v_model.wv[word]))
        similar_words = w2v_model.wv.most_similar(word, topn=5)
        print("\nMost similar words to '{}':".format(word))
        for similar_word, similarity in similar_words:
            print("Word: {}, Similarity: {}".format(similar_word, similarity))
    else:
        print("\nWord '{}' not found in the model vocabulary.".format(word))

    # Word similarity
    word_pairs = [('machine', 'learning'), ('text', 'processing'), ('love', 'fun')]
    print("\nWord Similarities:")
    for w1, w2 in word_pairs:
        if w1 in w2v_model.wv and w2 in w2v_model.wv:
            similarity = w2v_model.wv.similarity(w1, w2)
            print("'{}' and '{}' similarity: {}".format(w1, w2, similarity))
        else:
            print("One or both words '{}' and '{}' are not in the vocabulary.".format(w1, w2))

if __name__ == "__main__":
    main()
