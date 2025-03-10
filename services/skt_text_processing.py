import re
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import numpy as np
from models.skt_kobert import get_sentence_embedding

# 기존 카테고리 저장소 (초기 없음)
category_embeddings: Dict[str, List[np.ndarray]] = {}

# # URL 정규 표현식 패턴
# url_pattern = re.compile(r'https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(/\S*)?')
#
#
# # 문장에서 URL을 추출하고 문장 내에서 분리하는 함수
# def extract_urls_from_sentences(sentences: List[str]):
#     processed_sentences = []
#
#     for sentence in sentences:
#         urls = url_pattern.findall(sentence)  # URL 추출
#         text_without_urls = url_pattern.sub('', sentence).strip()  # URL 제거된 텍스트
#
#         processed_sentences.append({
#             "original_sentence": sentence,  # 원본 문장
#             "text_part": text_without_urls if text_without_urls else None,  # URL을 제외한 텍스트
#             "url_part": urls if urls else None  # 감지된 URL 리스트
#         })
#
#     return processed_sentences
# URL 정규 표현식 (전체 URL 추출)
url_pattern = re.compile(r'https?://[^\s]+')
# url_pattern = re.compile(r'https?://[a-zA-Z0-9./?=&_%:-]+')
# url_pattern = re.compile(r'https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(/\S*)?')
# 문장에서 URL을 추출하고 문장 내에서 분리하는 함수
def extract_urls_from_sentences(sentences: List[str]):
    processed_sentences = []

    for sentence in sentences:
        urls = url_pattern.findall(sentence)  # 문장에서 URL 추출
        text_without_urls = url_pattern.sub('', sentence).strip()  # URL을 제거한 텍스트

        processed_sentences.append({
            "original_sentence": sentence,  # 원본 문장
            "text_part": text_without_urls if text_without_urls else None,  # URL을 제외한 텍스트
            "url_part": urls if urls else None  # 감지된 URL 리스트 (정확한 URL 전체 포함)
        })

    return processed_sentences

# 문장 분류 함수 (유사 카테고리 확인 및 생성)
def classify_or_create_category(sentences: List[str], threshold: float = 0.7):
    global category_embeddings
    categorized_sentences = {}

    # 문장에서 URL을 분리하여 저장
    processed_sentences = extract_urls_from_sentences(sentences)

    for data in processed_sentences:
        text_part = data["text_part"]  # URL을 제거한 텍스트
        original_sentence = data["original_sentence"]

        if not text_part:
            categorized_sentences.setdefault("Uncategorized", []).append(data)
            continue

        sentence_embedding = get_sentence_embedding(text_part)

        best_category = None
        best_similarity = 0

        # 기존 카테고리와 유사도 비교
        for category, embeddings in category_embeddings.items():
            similarities = cosine_similarity([sentence_embedding], embeddings).mean()
            if similarities > best_similarity:
                best_similarity = similarities
                best_category = category

        # 기존 카테고리 존재 시 추가, 없으면 새 카테고리 생성
        if best_similarity >= threshold and best_category:
            categorized_sentences.setdefault(best_category, []).append(data)
            category_embeddings[best_category].append(sentence_embedding)
        else:
            new_category = f"Category_{len(category_embeddings) + 1}"
            category_embeddings[new_category] = [sentence_embedding]
            categorized_sentences.setdefault(new_category, []).append(data)

    return categorized_sentences
