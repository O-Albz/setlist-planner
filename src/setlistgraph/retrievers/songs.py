
from __future__ import annotations
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleSongRetriever:
    """Lightweight TF-IDF retriever over the song catalog.

    Combines `themes`, `scripture_refs`, and `lyrics_blurb` into a bag-of-words field,
    builds a TF-IDF index, and returns top-k similar rows to a query.
    """
    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        df['__bag'] = (
            df['themes'].fillna('') + ' ' +
            df['scripture_refs'].fillna('') + ' ' +
            df['lyrics_blurb'].fillna('')
        )
        self.df = df
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.X = self.vectorizer.fit_transform(self.df['__bag'])

    def search(self, query: str, top_k: int = 25) -> pd.DataFrame:
        if not query or not query.strip():
            # if no query, return by simple popularity proxy (energy+youth) or just head
            out = self.df.assign(score=0.0).head(top_k)
            return out
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.X)[0]
        out = self.df.copy()
        out['score'] = sims
        out = out.sort_values('score', ascending=False).head(top_k)
        return out

def filter_by_gathering(df: pd.DataFrame, gathering: str) -> pd.DataFrame:
    g = (gathering or '').strip().lower()
    if g == 'youth':
        return df.sort_values(['youth_score', 'energy', 'score'], ascending=False)
    elif g == 'traditional':
        # tilt toward lower energy / singable keys
        return df.sort_values(['energy', 'score'], ascending=[True, False])
    else:
        # family/general: favor score then moderate energy
        return df.sort_values(['score', 'energy'], ascending=[False, True])
