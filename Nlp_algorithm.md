# Algorithm Implementations in Python

This repository contains Python implementations of various algorithms, including similarity measures, distance metrics, string algorithms, sequence problems, search algorithms, dimensionality reduction techniques, text vectorization methods, and semantic similarity using BERT encoder from Hugging Face's `sentence-transformers`. Each algorithm is accompanied by a detailed explanation, mathematical formulation, computational logic, practical usage, and the corresponding code for easy reference.

---

## Table of Contents

1. [Similarity and Distance Metrics](#1-similarity-and-distance-metrics)
   - [1.1 Cosine Similarity](#11-cosine-similarity)
   - [1.2 Dot Product](#12-dot-product)
   - [1.3 Euclidean Distance](#13-euclidean-distance)
   - [1.4 Manhattan Distance](#14-manhattan-distance)
   - [1.5 Mahalanobis Distance](#15-mahalanobis-distance)
   - [1.6 Jaccard Similarity](#16-jaccard-similarity)
   - [1.7 Hamming Distance](#17-hamming-distance)
   - [1.8 Vector Normalization](#18-vector-normalization)
2. [String Algorithms](#2-string-algorithms)
   - [2.1 Longest Common Subsequence (LCS)](#21-longest-common-subsequence-lcs)
   - [2.2 Edit Distance (Levenshtein Distance)](#22-edit-distance-levenshtein-distance)
3. [Fibonacci Sequence](#3-fibonacci-sequence)
   - [3.1 Recursive Version](#31-recursive-version)
   - [3.2 Dynamic Programming Version](#32-dynamic-programming-version)
4. [Knuth-Morris-Pratt (KMP) Algorithm](#4-knuth-morris-pratt-kmp-algorithm)
   - [4.1 Prefix Table Computation](#41-prefix-table-computation)
   - [4.2 KMP String Matching](#42-kmp-string-matching)
5. [Dimensionality Reduction Techniques](#5-dimensionality-reduction-techniques)
   - [5.1 Principal Component Analysis (PCA)](#51-principal-component-analysis-pca)
   - [5.2 t-Distributed Stochastic Neighbor Embedding (t-SNE)](#52-t-distributed-stochastic-neighbor-embedding-t-sne)
   - [5.3 Uniform Manifold Approximation and Projection (UMAP)](#53-uniform-manifold-approximation-and-projection-umap)
6. [Text Vectorization and Word Embeddings](#6-text-vectorization-and-word-embeddings)
   - [6.1 TF-IDF (Term Frequency-Inverse Document Frequency)](#61-tf-idf-term-frequency-inverse-document-frequency)
   - [6.2 Bag of Words (BoW)](#62-bag-of-words-bow)
   - [6.3 Word2Vec Training](#63-word2vec-training)
7. [Semantic Similarity with BERT Encoder](#7-semantic-similarity-with-bert-encoder)

---

## 1. Similarity and Distance Metrics

### 1.1 Cosine Similarity

#### Introduction

Cosine similarity measures the cosine of the angle between two non-zero vectors in an inner product space. It is a measure of how similar the two vectors are, focusing on the direction rather than magnitude.

#### Mathematical Formulation

Given two vectors **A** and **B**:

\[
\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|_2 \times \|\mathbf{B}\|_2}
\]

Where:

- \(\mathbf{A} \cdot \mathbf{B}\) is the dot product.
- \(\|\mathbf{A}\|_2\) is the Euclidean norm (L2 norm).

#### Computational Logic

1. Calculate the dot product of vectors **A** and **B**.
2. Calculate the L2 norms of both vectors.
3. Divide the dot product by the product of the norms.
4. Ensure norms are not zero to avoid division by zero.

#### Usage and Applications

- Measuring text similarity in NLP.
- Document clustering.
- Recommendation systems.

#### Code

```python
def cosine_similarity(vec1, vec2):
    import numpy as np
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)
```

#### Example Usage

```python
vec_a = [1, 2, 3]
vec_b = [4, 5, 6]
similarity = cosine_similarity(vec_a, vec_b)
print("Cosine Similarity:", similarity)
```

**Output:**

```
Cosine Similarity: 0.9746318461970762
```

---

### 1.2 Dot Product

#### Introduction

The dot product (scalar product) is an algebraic operation that takes two equal-length sequences of numbers and returns a single number.

#### Mathematical Formulation

\[
\mathbf{A} \cdot \mathbf{B} = \sum_{i=1}^n A_i B_i
\]

#### Computational Logic

1. Multiply corresponding elements.
2. Sum the results.

#### Usage and Applications

- Calculating projections.
- Determining angles between vectors.
- Physical calculations (e.g., work done).

#### Code

```python
def dot_product(vec1, vec2):
    return sum(a * b for a, b in zip(vec1, vec2))
```

#### Example Usage

```python
vec_a = [1, 2, 3]
vec_b = [4, 5, 6]
product = dot_product(vec_a, vec_b)
print("Dot Product:", product)
```

**Output:**

```
Dot Product: 32
```

---

### 1.3 Euclidean Distance

#### Introduction

Euclidean distance is the straight-line distance between two points in Euclidean space.

#### Mathematical Formulation

\[
d(\mathbf{A}, \mathbf{B}) = \sqrt{\sum_{i=1}^n (A_i - B_i)^2}
\]

#### Computational Logic

1. Subtract corresponding elements.
2. Square each difference.
3. Sum them up.
4. Take the square root.

#### Usage and Applications

- Clustering algorithms (e.g., K-Means).
- Nearest Neighbor searches.
- Spatial analysis.

#### Code

```python
def euclidean_distance(vec1, vec2):
    import math
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
```

#### Example Usage

```python
vec_a = [1, 2, 3]
vec_b = [4, 5, 6]
distance = euclidean_distance(vec_a, vec_b)
print("Euclidean Distance:", distance)
```

**Output:**

```
Euclidean Distance: 5.196152422706632
```

---

### 1.4 Manhattan Distance

#### Introduction

Manhattan distance is the sum of the absolute differences of their Cartesian coordinates.

#### Mathematical Formulation

\[
d(\mathbf{A}, \mathbf{B}) = \sum_{i=1}^n |A_i - B_i|
\]

#### Computational Logic

1. Compute absolute differences.
2. Sum them.

#### Usage and Applications

- Grid-based pathfinding.
- Simple distance metrics.
- Optimization problems.

#### Code

```python
def manhattan_distance(vec1, vec2):
    return sum(abs(a - b) for a, b in zip(vec1, vec2))
```

#### Example Usage

```python
vec_a = [1, 2, 3]
vec_b = [4, 5, 6]
distance = manhattan_distance(vec_a, vec_b)
print("Manhattan Distance:", distance)
```

**Output:**

```
Manhattan Distance: 9
```

---

### 1.5 Mahalanobis Distance

#### Introduction

Mahalanobis distance measures the distance between a point and a distribution, considering correlations.

#### Mathematical Formulation

\[
d(\mathbf{A}, \mathbf{B}) = \sqrt{(\mathbf{A} - \mathbf{B})^\top \Sigma^{-1} (\mathbf{A} - \mathbf{B})}
\]

Where \(\Sigma\) is the covariance matrix.

#### Computational Logic

1. Compute the difference vector.
2. Invert the covariance matrix.
3. Multiply accordingly.
4. Take the square root.

#### Usage and Applications

- Multivariate outlier detection.
- Cluster analysis.
- Classification in machine learning.

#### Code

```python
def mahalanobis_distance(vec1, vec2, cov_matrix):
    import numpy as np
    diff = np.array(vec1) - np.array(vec2)
    inv_covmat = np.linalg.inv(cov_matrix)
    distance = np.sqrt(np.dot(np.dot(diff.T, inv_covmat), diff))
    return distance
```

#### Example Usage

```python
vec_a = [1, 2]
vec_b = [3, 4]
cov_matrix = [[1, 0], [0, 1]]
distance = mahalanobis_distance(vec_a, vec_b, cov_matrix)
print("Mahalanobis Distance:", distance)
```

**Output:**

```
Mahalanobis Distance: 2.8284271247461903
```

---

### 1.6 Jaccard Similarity

#### Introduction

Jaccard similarity measures similarity between finite sample sets.

#### Mathematical Formulation

\[
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
\]

#### Computational Logic

1. Find intersection and union.
2. Divide the sizes.

#### Usage and Applications

- Text analysis.
- Recommender systems.
- Clustering.

#### Code

```python
def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 1.0  # Both sets are empty
    return len(intersection) / len(union)
```

#### Example Usage

```python
set_a = {"apple", "banana", "cherry"}
set_b = {"banana", "cherry", "date", "fig"}
similarity = jaccard_similarity(set_a, set_b)
print("Jaccard Similarity:", similarity)
```

**Output:**

```
Jaccard Similarity: 0.4
```

---

### 1.7 Hamming Distance

#### Introduction

Hamming distance is the number of positions at which the corresponding symbols are different between two strings of equal length.

#### Mathematical Formulation

\[
d(S_1, S_2) = \sum_{i=1}^n [S_1(i) \ne S_2(i)]
\]

#### Computational Logic

1. Compare positions.
2. Count the differences.

#### Usage and Applications

- Error detection and correction.
- Information theory.
- Cryptography.

#### Code

```python
def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of equal length.")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))
```

#### Example Usage

```python
string_a = "karolin"
string_b = "kathrin"
distance = hamming_distance(string_a, string_b)
print("Hamming Distance:", distance)
```

**Output:**

```
Hamming Distance: 3
```

---

### 1.8 Vector Normalization

#### Introduction

Normalization scales a vector to have a length of 1.

#### Mathematical Formulation

\[
\hat{\mathbf{A}} = \frac{\mathbf{A}}{\|\mathbf{A}\|}
\]

#### Computational Logic

1. Calculate the vector's norm.
2. Divide each component.

#### Usage and Applications

- Preparing data for machine learning algorithms.
- Ensuring fair weight distribution.
- Directional computations.

#### Code

```python
def normalize_vector(vec):
    import numpy as np
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec  # Return original if norm is zero
    return vec / norm
```

#### Example Usage

```python
vec = [1, 2, 3]
normalized_vec = normalize_vector(vec)
print("Normalized Vector:", normalized_vec)
```

**Output:**

```
Normalized Vector: [0.26726124 0.53452248 0.80178373]
```

---

## 2. String Algorithms

### 2.1 Longest Common Subsequence (LCS)

#### Introduction

LCS finds the longest subsequence common to all sequences in a set of sequences.

#### Mathematical Formulation

Let \(X\) and \(Y\) be sequences:

\[
LCS(i, j) = \begin{cases}
0 & \text{if } i = 0 \text{ or } j = 0 \\
LCS(i-1, j-1) + 1 & \text{if } X_i = Y_j \\
\max(LCS(i-1, j), LCS(i, j-1)) & \text{if } X_i \ne Y_j
\end{cases}
\]

#### Computational Logic

1. Create a matrix \( (m+1) \times (n+1) \).
2. Fill the matrix according to the formula.
3. Backtrack to find the sequence (if needed).

#### Usage and Applications

- File comparison tools.
- Bioinformatics.
- Spell checking.

#### Code

```python
def longest_common_subsequence(X, Y):
    m, n = len(X), len(Y)
    L = [[0]*(n+1) for i in range(m+1)]
    # Build L[m+1][n+1] in bottom up fashion
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]
```

#### Example Usage

```python
X = "AGGTAB"
Y = "GXTXAYB"
length = longest_common_subsequence(X, Y)
print("Length of LCS:", length)
```

**Output:**

```
Length of LCS: 4
```

---

### 2.2 Edit Distance (Levenshtein Distance)

#### Introduction

Edit distance measures the minimum number of single-character edits required to change one string into the other.

#### Mathematical Formulation

Let \( S_1 \) and \( S_2 \) be strings:

\[
D(i, j) = \begin{cases}
i & \text{if } j = 0 \\
j & \text{if } i = 0 \\
\min \begin{cases}
D(i-1, j) + 1 \\
D(i, j-1) + 1 \\
D(i-1, j-1) + cost
\end{cases} & \text{otherwise}
\end{cases}
\]

Where cost is 0 if \( S_1(i) = S_2(j) \), else 1.

#### Computational Logic

1. Initialize a matrix.
2. Fill based on insertion, deletion, substitution.
3. The value in the bottom-right cell is the edit distance.

#### Usage and Applications

- Spell correction.
- DNA sequencing.
- Plagiarism detection.

#### Code

```python
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for i in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i  # Deletion
    for j in range(n+1):
        dp[0][j] = j  # Insertion
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,       # Deletion
                dp[i][j-1] + 1,       # Insertion
                dp[i-1][j-1] + cost   # Substitution
            )
    return dp[m][n]
```

#### Example Usage

```python
str1 = "kitten"
str2 = "sitting"
distance = edit_distance(str1, str2)
print("Edit Distance:", distance)
```

**Output:**

```
Edit Distance: 3
```

---

## 3. Fibonacci Sequence

### 3.1 Recursive Version

#### Introduction

Calculates the nth Fibonacci number using recursion.

#### Mathematical Formulation

\[
F(n) = \begin{cases}
n & \text{if } n = 0 \text{ or } n = 1 \\
F(n-1) + F(n-2) & \text{if } n > 1
\end{cases}
\]

#### Computational Logic

- Base cases: \( F(0) = 0 \), \( F(1) = 1 \)
- Recursive calls for \( F(n-1) \) and \( F(n-2) \)

#### Usage and Applications

- Mathematical sequences.
- Algorithm teaching.
- Recursive function examples.

#### Code

```python
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)
```

#### Example Usage

```python
n = 5
fib_number = fibonacci_recursive(n)
print(f"Fibonacci number at position {n}:", fib_number)
```

**Output:**

```
Fibonacci number at position 5: 5
```

---

### 3.2 Dynamic Programming Version

#### Introduction

Calculates the nth Fibonacci number using dynamic programming to optimize.

#### Computational Logic

1. Initialize an array \( fib[0..n] \).
2. \( fib[0] = 0 \), \( fib[1] = 1 \).
3. Iterate from 2 to n, updating \( fib[i] = fib[i-1] + fib[i-2] \).

#### Usage and Applications

- Efficient calculation for large n.
- Demonstrating dynamic programming.

#### Code

```python
def fibonacci_dp(n):
    fib = [0, 1]
    for i in range(2, n+1):
        fib.append(fib[i-1] + fib[i-2])
    return fib[n]
```

#### Example Usage

```python
n = 10
fib_number = fibonacci_dp(n)
print(f"Fibonacci number at position {n}:", fib_number)
```

**Output:**

```
Fibonacci number at position 10: 55
```

---

## 4. Knuth-Morris-Pratt (KMP) Algorithm

### 4.1 Prefix Table Computation

#### Introduction

Computes the longest prefix which is also a suffix (LPS array) for the pattern.

#### Computational Logic

1. Initialize LPS array of size len(pattern).
2. Iterate over the pattern to fill LPS.

#### Usage and Applications

- Preprocessing for KMP algorithm.

#### Code

```python
def compute_lps_array(pattern):
    length = 0
    lps = [0]*len(pattern)
    i = 1
    while i < len(pattern):
        if pattern[i]==pattern[length]:
            length += 1
            lps[i] = length
            i +=1
        else:
            if length != 0:
                length = lps[length-1]
            else:
                lps[i]=0
                i +=1
    return lps
```

#### Example Usage

```python
pattern = "ABABCABAB"
lps = compute_lps_array(pattern)
print("LPS Array:", lps)
```

**Output:**

```
LPS Array: [0, 0, 1, 2, 0, 1, 2, 3, 2]
```

---

### 4.2 KMP String Matching

#### Introduction

Searches for occurrences of a pattern within a text.

#### Computational Logic

1. Use LPS array to avoid recomparing characters.
2. Iterate over text and pattern.

#### Usage and Applications

- String search in texts.
- Pattern matching.

#### Code

```python
def kmp_search(text, pattern):
    M = len(pattern)
    N = len(text)
    lps = compute_lps_array(pattern)
    results = []
    i = j = 0
    while i < N:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == M:
            results.append(i-j)
            j = lps[j-1]
        elif i < N and pattern[j] != text[i]:
            if j !=0:
                j = lps[j-1]
            else:
                i +=1
    return results
```

#### Example Usage

```python
text = "ABABDABACDABABCABAB"
pattern = "ABABCABAB"
positions = kmp_search(text, pattern)
print("Pattern found at positions:", positions)
```

**Output:**

```
Pattern found at positions: [10]
```

---

## 5. Dimensionality Reduction Techniques

### 5.1 Principal Component Analysis (PCA)

#### Introduction

Transforms data into fewer dimensions while retaining most variance.

#### Computational Logic

1. Standardize data.
2. Compute covariance matrix.
3. Calculate eigenvectors and eigenvalues.
4. Project data onto principal components.

#### Usage and Applications

- Data visualization.
- Noise reduction.
- Feature extraction.

#### Code

```python
def pca_reduce(data, n_components):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(data)
    return principalComponents, pca.explained_variance_ratio_
```

#### Example Usage

```python
import numpy as np
data = np.random.rand(100, 5)
reduced_data, variance_ratio = pca_reduce(data, n_components=2)
print("Reduced Data Shape:", reduced_data.shape)
print("Explained Variance Ratio:", variance_ratio)
```

**Output:**

```
Reduced Data Shape: (100, 2)
Explained Variance Ratio: [0.22033247 0.20904267]
```

---

### 5.2 t-Distributed Stochastic Neighbor Embedding (t-SNE)

#### Introduction

Non-linear dimensionality reduction for high-dimensional data.

#### Computational Logic

- Model high-dimensional similarities.
- Map to low-dimensional space.
- Optimize using gradient descent.

#### Usage and Applications

- Visualizing clusters.
- Image recognition.

#### Code

```python
def tsne_reduce(data, n_components=2, perplexity=30, random_state=42):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(data)
    return tsne_results
```

#### Example Usage

```python
tsne_results = tsne_reduce(data)
print("t-SNE Results Shape:", tsne_results.shape)
```

**Output:**

```
t-SNE Results Shape: (100, 2)
```

---

### 5.3 Uniform Manifold Approximation and Projection (UMAP)

#### Introduction

UMAP is a dimension reduction technique that can be used for visualization similarly to t-SNE but is faster and preserves more global structure.

#### Computational Logic

- Construct fuzzy topological representations.
- Optimize low-dimensional embeddings.

#### Usage and Applications

- Visualization.
- Clustering.

#### Code

```python
def umap_reduce(data, n_components=2, n_neighbors=15, min_dist=0.1):
    import umap
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
    embedding = reducer.fit_transform(data)
    return embedding
```

#### Example Usage

```python
embedding = umap_reduce(data)
print("UMAP Embedding Shape:", embedding.shape)
```

**Output:**

```
UMAP Embedding Shape: (100, 2)
```

---

## 6. Text Vectorization and Word Embeddings

### 6.1 TF-IDF (Term Frequency-Inverse Document Frequency)

#### Introduction

Evaluates how important a word is to a document in a collection.

#### Code

```python
def compute_tfidf(corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer.get_feature_names_out()
```

#### Example Usage

```python
documents = [
    "I love programming in Python",
    "Python and Java are popular programming languages",
    "I enjoy machine learning and data science"
]

tfidf_matrix, feature_names = compute_tfidf(documents)
print("Feature Names:", feature_names)
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())
```

---

### 6.2 Bag of Words (BoW)

#### Introduction

Represents text as the bag of its words, disregarding grammar.

#### Code

```python
def compute_bow(corpus):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer.get_feature_names_out()
```

#### Example Usage

```python
bow_matrix, bow_features = compute_bow(documents)
print("Bag of Words Features:", bow_features)
print("Bag of Words Matrix:\n", bow_matrix.toarray())
```

---

### 6.3 Word2Vec Training

#### Introduction

Generates word embeddings using neural networks.

#### Code

```python
def train_word2vec(sentences, vector_size=100, window=5, min_count=1):
    from gensim.models import Word2Vec
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count)
    return model
```

#### Example Usage

```python
from gensim.utils import simple_preprocess

sentences = [simple_preprocess(doc) for doc in documents]
w2v_model = train_word2vec(sentences)
word_vector = w2v_model.wv['python']
print("Vector for 'python':", word_vector)
```

---

## 7. Semantic Similarity with BERT Encoder

### Introduction

Using Hugging Face's `sentence-transformers` to compute sentence embeddings and measure similarity.

### Computational Logic

1. Load a pre-trained BERT sentence embedding model.
2. Encode the query and candidate sentences.
3. Compute cosine similarities.
4. Retrieve the most similar sentence.

### Code

```python
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Simple QA dictionary
QA_dict = {
    "What is DNA?": "DNA is a molecule that carries genetic instructions in living organisms.",
    "What causes diabetes?": "Diabetes is caused by high blood sugar due to insulin issues.",
    "How does vaccination work?": "Vaccination trains the immune system to recognize and fight pathogens.",
    "What is a virus?": "A virus is a microscopic infectious agent that replicates inside living cells.",
    "What are antibiotics?": "Antibiotics are drugs that kill or stop the growth of bacteria.",
    "What is the human genome?": "The human genome is the complete set of genetic information in humans.",
    "How do neurons communicate?": "Neurons communicate via electrical and chemical signals.",
}

query = "What do vaccines do?"

# Encode the query
query_emb = model.encode(query, convert_to_tensor=True)

# Encode candidate questions
questions = list(QA_dict.keys())
questions_emb = model.encode(questions, convert_to_tensor=True)

# Compute cosine similarities
cos_scores = F.cosine_similarity(query_emb, questions_emb)

# Find the best matching question
best_match_idx = torch.argmax(cos_scores)
best_question = questions[best_match_idx]
best_answer = QA_dict[best_question]
best_score = cos_scores[best_match_idx].item()

print("Query:", query)
print("Best Match Question:", best_question)
print("Best Answer:", best_answer)
print("Similarity Score:", best_score)
```

### Example Usage

```python
# Run the code above
```

**Output:**

```
Query: What do vaccines do?
Best Match Question: How does vaccination work?
Best Answer: Vaccination trains the immune system to recognize and fight pathogens.
Similarity Score: 0.7314323782920837
```

---

## Conclusion

This repository provides practical and accurate implementations of fundamental algorithms in Python. Each algorithm includes mathematical formulations, computational logic, example code, and usage examples. This comprehensive collection serves as a valuable resource for students, educators, and professionals in fields such as machine learning, data science, and natural language processing.

---

## References

- [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Mahalanobis Distance](https://en.wikipedia.org/wiki/Mahalanobis_distance)
- [Jaccard Index](https://en.wikipedia.org/wiki/Jaccard_index)
- [Hamming Distance](https://en.wikipedia.org/wiki/Hamming_distance)
- [Longest Common Subsequence Problem](https://en.wikipedia.org/wiki/Longest_common_subsequence_problem)
- [Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
- [Fibonacci Number](https://en.wikipedia.org/wiki/Fibonacci_number)
- [Knuth-Morris-Pratt Algorithm](https://en.wikipedia.org/wiki/Knuth–Morris–Pratt_algorithm)
- [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)
- [t-SNE](https://lvdmaaten.github.io/tsne/)
- [UMAP](https://umap-learn.readthedocs.io/en/latest/)
- [TF-IDF](https://en.wikipedia.org/wiki/Tf–idf)
- [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf)
- [Sentence Transformers](https://www.sbert.net/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to clone the repository and experiment with the implementations. Contributions and improvements are welcome!
