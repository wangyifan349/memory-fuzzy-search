#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import jieba
import unicodedata
from whoosh import index as whoosh_index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import FunctionAnalyzer
from whoosh.qparser import MultifieldParser

# --------------------------------------------------------------------------------
# 1. TOKENIZER: remove all Unicode punctuation, then segment text via jieba or whitespace
#    分词器：去除所有 Unicode 标点，再用 jieba（中文）或空白分词（其他语言）
# --------------------------------------------------------------------------------
def remove_unicode_punctuation_and_tokenize(text):
    """
    English:
      Strip out any Unicode punctuation, then yield tokens.
      For Chinese, jieba.cut splits words; for other languages, splits on whitespace.
    中文：
      去除所有 Unicode 标点符号后分词。中文用 jieba.cut，其他语言按空白分词。
    """
    # Build a new string without Unicode punctuation
    cleaned_characters = []
    for character in text:
        # The Unicode category 'P*' stands for all punctuation
        if not unicodedata.category(character).startswith("P"):
            cleaned_characters.append(character)
    cleaned_text = "".join(cleaned_characters)

    # If the text contains CJK characters, use jieba; else simple whitespace split
    # 判断是否含有中文/日文/韩文字符（CJK Unified Ideographs）
    if any('\u4e00' <= ch <= '\u9fff' for ch in cleaned_text):
        # Chinese (and similar CJK) text → use jieba
        for word in jieba.cut(cleaned_text):
            token = word.strip()
            if token:
                yield token
    else:
        # Non-CJK text → split on whitespace
        for token in cleaned_text.split():
            token = token.strip()
            if token:
                yield token

# --------------------------------------------------------------------------------
# 2. DEFINE SCHEMA: path (unique ID), title (boost=2.0), content (boost=1.0)
#    定义索引结构：路径（唯一ID），标题（权重2.0），内容（权重1.0）
# --------------------------------------------------------------------------------
def make_schema():
    """
    English:
      Create a Whoosh Schema with:
        - path: stored unique identifier (file path)
        - title: text field, higher boost (2.0)
        - content: text field, normal boost (1.0)
    中文：
      构造 Whoosh Schema：
        - path：存储且唯一的标识符（文件路径）
        - title：文本字段，权重提升到 2.0
        - content：文本字段，默认权重 1.0
    """
    analyzer = FunctionAnalyzer(remove_unicode_punctuation_and_tokenize)
    return Schema(
        path=ID(stored=True, unique=True),
        title=TEXT(stored=True, analyzer=analyzer, field_boost=2.0),
        content=TEXT(stored=True, analyzer=analyzer, field_boost=1.0)
    )

# --------------------------------------------------------------------------------
# 3. ENSURE INDEX: create or open an index at given directory, index all .txt in docs
#    建立或打开索引：在 index_directory 下创建或打开索引，并索引 docs_directory 中的所有 .txt
# --------------------------------------------------------------------------------
def ensure_search_index(index_directory="indexdir", docs_directory="docs"):
    """
    English:
      If index_directory does not exist, create it and build a new index
      from all .txt files under docs_directory. Otherwise open existing index.
    中文：
      若 index_directory 不存在，则创建并用 docs_directory 下所有 .txt 文件建立索引；
      否则直接打开已有索引。
    """
    if not os.path.exists(index_directory):
        os.makedirs(index_directory)

    if whoosh_index.exists_in(index_directory):
        # Open existing index
        return whoosh_index.open_dir(index_directory)
    else:
        # Create new index
        schema = make_schema()
        new_index = whoosh_index.create_in(index_directory, schema)
        writer = new_index.writer()

        # Iterate all .txt files and add to index
        for filename in os.listdir(docs_directory):
            if not filename.lower().endswith(".txt"):
                continue
            file_path = os.path.join(docs_directory, filename)
            with open(file_path, encoding="utf-8") as text_file:
                all_lines = text_file.readlines()
            if not all_lines:
                continue
            document_title = all_lines[0].strip()
            document_content = "".join(all_lines[1:]).strip()
            writer.add_document(
                path=file_path,
                title=document_title,
                content=document_content
            )
        writer.commit()
        return new_index

# --------------------------------------------------------------------------------
# 4. SEARCH FUNCTION: run a query against both title and content with optional boosts
#    搜索函数：在标题和内容字段上查询，支持动态字段权重
# --------------------------------------------------------------------------------
def perform_search(query_string,
                   search_index,
                   max_results=10,
                   field_boosts=None):
    """
    English:
      Search the given Whoosh index for query_string, returning up to max_results.
      field_boosts: dict e.g. {"title": 3.0, "content":1.0}
    中文：
      在指定的 Whoosh 索引上执行查询 query_string，返回最多 max_results 条记录。
      field_boosts 可传入动态字段权重，如 {"title":3.0, "content":1.0}
    """
    parser = MultifieldParser(
        ["title", "content"],
        schema=search_index.schema,
        fieldboosts=field_boosts or {}
    )
    parsed_query = parser.parse(query_string)
    results_summary = []

    with search_index.searcher() as searcher:
        hits = searcher.search(parsed_query, limit=max_results)
        for hit in hits:
            results_summary.append({
                "path":    hit["path"],
                "title":   hit["title"],
                "score":   round(hit.score, 4),
                "snippet": hit.highlights("content", top=2)
            })
    return results_summary

# --------------------------------------------------------------------------------
# 5. INITIALIZE INDEX & START INTERACTIVE LOOP
#    初始化索引，并进入持续查询（while True）交互模式
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # Configuration — 配置项
    DOCUMENTS_DIRECTORY = "docs"        # folder with .txt files
    INDEX_DIRECTORY     = "indexdir"    # where the Whoosh index is stored
    MAX_HITS            = 5             # maximum results to display
    # default field boosts: titles more important than content
    DEFAULT_FIELD_BOOSTS = {"title": 2.5, "content": 1.0}

    # 1) Build or open the index once
    search_index_instance = ensure_search_index(
        index_directory=INDEX_DIRECTORY,
        docs_directory=DOCUMENTS_DIRECTORY
    )

    # 2) Enter interactive loop
    print("Local Full-Text Search ready. 输入查询关键词，输入 exit 或 空行 退出。")
    while True:
        user_input = input(">>> ").strip()
        if user_input.lower() == "exit" or not user_input:
            print("Exiting search. 退出检索。")
            break

        # Perform search
        search_results = perform_search(
            query_string=user_input,
            search_index=search_index_instance,
            max_results=MAX_HITS,
            field_boosts=DEFAULT_FIELD_BOOSTS
        )

        # Display results
        if not search_results:
            print("  No matches found. 未找到匹配结果。\n")
        else:
            print(f"  Query: {user_input} — {len(search_results)} hits.\n"
                  f"  查询：{user_input} — 共 {len(search_results)} 条结果。\n")
            for rank, item in enumerate(search_results, start=1):
                print(f"  [{rank}] Title: {item['title']}")
                print(f"       路径: {item['path']}")
                print(f"       Score: {item['score']}，得分")
                print(f"       Snippet: {item['snippet']}\n")
            print("-" * 60)
