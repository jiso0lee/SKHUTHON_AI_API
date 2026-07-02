from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="청춘일기장 추천 서비스")


class DiaryRecommender:
    KEEP_TAGS = ("NNG", "NNP", "VA", "VV", "XR")
    STOPWORDS = {"오늘", "하", "있", "되", "너무", "또", "거", "것", "수", "때"}

    def __init__(self):
        self.kiwi = Kiwi()
        self.diaries = {}          
        self.ids = []             
        self.tfidf = None
        self.vectorizer = None

    def _tokenize(self, text):
        toks = self.kiwi.tokenize(text)
        return " ".join(t.form for t in toks
                        if t.tag in self.KEEP_TAGS
                        and t.form not in self.STOPWORDS and len(t.form) > 1)

    def _rebuild(self):
        if not self.diaries:
            self.tfidf = self.vectorizer = None
            self.ids = []
            return
        self.ids = list(self.diaries.keys())
        corpus = [self._tokenize(self.diaries[i]["text"]) for i in self.ids]
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        self.tfidf = self.vectorizer.fit_transform(corpus)

    def upsert(self, diary_id, user_id, text):
        self.diaries[diary_id] = {"user_id": user_id, "text": text}
        self._rebuild()

    def delete(self, diary_id):
        self.diaries.pop(diary_id, None)
        self._rebuild()

    def reindex(self, items):
        self.diaries = {d["diary_id"]: {"user_id": d["user_id"], "text": d["text"]}
                        for d in items}
        self._rebuild()

    def similar_diaries(self, diary_id, top_k=5):
        if diary_id not in self.diaries:
            raise KeyError(diary_id)
        row = self.ids.index(diary_id)
        sims = cosine_similarity(self.tfidf[row], self.tfidf).flatten()
        sims[row] = -1
        order = np.argsort(sims)[::-1][:top_k]
        return [{"diary_id": int(self.ids[i]), "score": round(float(sims[i]), 3)}
                for i in order if sims[i] > 0]

    def recommend_for_user(self, user_id, top_k=5):
        user_ids = {d["user_id"] for d in self.diaries.values()}
        if user_id not in user_ids:
            raise KeyError(user_id)

        def profile(uid):
            rows = [self.ids.index(did) for did, d in self.diaries.items()
                    if d["user_id"] == uid]
            return np.asarray(self.tfidf[rows].mean(axis=0))

        target = profile(user_id)
        sims = {uid: float(cosine_similarity(target, profile(uid))[0, 0])
                for uid in user_ids if uid != user_id}
        ranked = sorted(sims.items(), key=lambda t: t[1], reverse=True)

        recs = []
        for uid, score in ranked:
            for did, d in self.diaries.items():
                if d["user_id"] == uid:
                    recs.append({"diary_id": int(did), "user_id": int(uid),
                                 "score": round(score, 3)})
        return recs[:top_k]

    def recommend_by_texts(self, base_texts, exclude_user_ids=(), top_k=5):
        if self.vectorizer is None:
            raise KeyError("색인이 비어있습니다 (먼저 일기를 색인하세요)")
        joined = " ".join(self._tokenize(t) for t in base_texts)
        base_vec = self.vectorizer.transform([joined])

        sims = cosine_similarity(base_vec, self.tfidf).flatten()
        order = np.argsort(sims)[::-1]

        exclude = set(exclude_user_ids)
        results = []
        for i in order:
            did = self.ids[i]
            d = self.diaries[did]
            if d["user_id"] in exclude:      
                continue
            if sims[i] <= 0.1:                 
                continue
            results.append({"diary_id": int(did), "user_id": int(d["user_id"]),
                            "score": round(float(sims[i]), 3)})
            if len(results) >= top_k:
                break
        return results

engine = DiaryRecommender()

class DiaryIn(BaseModel):
    id: int
    memberId: int
    title: str = ""
    content: str = ""

    @property
    def text(self) -> str:
        return f"{self.title} {self.content}".strip()

@app.post("/diaries")
def upsert_diary(d: DiaryIn):
    engine.upsert(d.id, d.memberId, d.text)
    return {"status": "indexed", "diary_id": d.id}


@app.delete("/diaries/{diary_id}")
def delete_diary(diary_id: int):
    engine.delete(diary_id)
    return {"status": "deleted", "diary_id": diary_id}


@app.post("/reindex")
def reindex(items: list[DiaryIn]):
    mapped = [{"diary_id": d.id, "user_id": d.memberId, "text": d.text}
              for d in items]
    engine.reindex(mapped)
    return {"status": "reindexed", "count": len(items)}


@app.get("/diaries/{diary_id}/similar")
def similar(diary_id: int, top_k: int = 5):
    try:
        return {"diary_id": diary_id, "results": engine.similar_diaries(diary_id, top_k)}
    except KeyError:
        raise HTTPException(404, "색인에 없는 일기입니다 (먼저 POST /diaries 필요)")


@app.get("/users/{user_id}/recommend")
def recommend(user_id: int, top_k: int = 5):
    try:
        return {"user_id": user_id, "results": engine.recommend_for_user(user_id, top_k)}
    except KeyError:
        raise HTTPException(404, "색인에 해당 유저의 일기가 없습니다")


class RecommendRequest(BaseModel):
    diaries: list[DiaryIn]     
    top_k: int = 5


@app.post("/recommend")
def recommend_by_recent(req: RecommendRequest):
    if not req.diaries:
        raise HTTPException(400, "기준 일기(diaries)가 비어있습니다")
    base_texts = [d.text for d in req.diaries]
    my_user_ids = {d.memberId for d in req.diaries}   
    try:
        results = engine.recommend_by_texts(base_texts, my_user_ids, req.top_k)
        return {"based_on": [d.id for d in req.diaries], "results": results}
    except KeyError as e:
        raise HTTPException(404, str(e))