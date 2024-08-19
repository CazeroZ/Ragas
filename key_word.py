import spacy
from spacy.tokens import DocBin
from spacy.language import Language
from spacy.pipeline import TextCategorizer
from spacy_transformers import Transformer

def extract_keywords_spacy(text, top_n=5):
    nlp = spacy.load("en_core_web_trf")  # 使用 spaCy 的 Transformer 模型
    doc = nlp(text)
    
    # 提取名词和名词短语作为关键词
    keywords = [chunk.text for chunk in doc.noun_chunks][:top_n]
    return keywords

# 示例使用
text = """
The history of natural language processing (NLP) generally started in the 1950s, although work can be found from earlier periods. In 1950, Alan Turing published an article titled "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence. The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English. The authors claimed that within three or five years, machine translation would be a solved problem. However, real progress was much slower, and after the ALPAC report in 1966, which found that ten-year-long research had failed to fulfill the expectations, funding for machine translation was dramatically reduced. Little further research in machine translation was conducted until the late 1980s when the first statistical machine translation systems were developed.
"""
keywords = extract_keywords_spacy(text, top_n=5)
print("Keywords:", keywords)
