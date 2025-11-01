# src/text_utils.py
from collections import Counter
import re
from wordcloud import WordCloud

def simple_word_counts(texts, topn=50):
    tokens = []
    for t in texts.dropna().astype(str):
        tokens += re.findall(r'\w+', t.lower())
    return Counter(tokens).most_common(topn)

def make_wordcloud(texts):
    wc = WordCloud(width=800, height=400).generate(' '.join(texts.dropna().astype(str)))
    return wc
