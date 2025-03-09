import re

def separate_text_and_urls(text):
    # URL 정규 표현식 패턴
    url_pattern = re.compile(
        r'(https?://[^\s]+)'  # http:// 또는 https://로 시작하는 문자열 추출
    )

    # URL 추출
    urls = url_pattern.findall(text)

    # URL을 제외한 일반 텍스트 추출
    text_without_urls = url_pattern.sub('', text).strip()

    return text_without_urls, urls

# 테스트
sample_text = "이 영화는 너무 재미있어요! 예매는 http://ticket.com에서 할 수 있어요."
text, urls = separate_text_and_urls(sample_text)

print("📌 일반 텍스트:", text)
print("🔗 추출된 URL:", urls)
