"""
청춘잇다 - AI 응원 메시지 생성 서버 (OpenAI GPT)
POST /generate-mood-message

실행: uvicorn title_recommend:app --host 0.0.0.0 --port 8000
환경변수: OPENAI_API_KEY 필요
"""
import os
import json
import time
import logging

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mood-message")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = "gpt-4o-mini"

client = OpenAI(
    api_key=OPENAI_API_KEY or "not-set",
    timeout=4.0,
    max_retries=0,
)

router = APIRouter(prefix="/title", tags=["Title"])

AGE_TRAITS = {
    "고등학생": {
        "특성": "학업 스트레스, 또래 관계, 진로 고민, 자아 탐색",
        "말투": "친구처럼 편하고 따뜻하게, 존댓말 사용",
        "공감포인트": "시험, 친구관계, 미래불안, 부모님과의 갈등",
    },
    "20대 초반": {
        "특성": "대학생활, 자유와 불안, 아르바이트, 새로운 만남",
        "말투": "동등하게 공감하며, 가볍고 따뜻하게",
        "공감포인트": "진로탐색, 인간관계, 돈걱정, 미래막막함",
    },
    "20대 중반": {
        "특성": "취업 준비, 비교와 불안, 자존감 하락, 독립",
        "말투": "진심 어린 공감과 응원, 너무 가볍지 않게",
        "공감포인트": "취준 압박, 남들과 비교, 자존감, 미래불안",
    },
    "20대 후반~30대 초반": {
        "특성": "직장생활, 커리어 고민, 결혼 압박, 책임감",
        "말투": "성숙하고 따뜻하게, 존중하는 톤으로",
        "공감포인트": "직장스트레스, 번아웃, 인간관계, 미래설계",
    },
    "30대 중반 이상": {
        "특성": "가정과 커리어 균형, 번아웃, 건강, 인생 재점검",
        "말투": "깊이 있게 공감하며, 따뜻하고 위로하는 톤",
        "공감포인트": "육아, 직장, 건강, 나를 잃어가는 느낌",
    },
}

EMOTION_GUIDE = {
    100: "매우 긍정적인 하루 → 이 에너지를 오래 간직하도록 응원",
    75: "긍정적인 하루 → 잘 보낸 오늘을 칭찬하고 내일도 응원",
    50: "보통인 하루 → 평범한 날도 소중하다고 위로, 쉬어가도 된다고",
    25: "부정적인 하루 → 힘든 감정 공감하고 따뜻하게 다독임",
    0: "매우 힘든 하루 → 깊이 공감하고 쉬어도 된다고 위로",
}

CACHE_TTL = 3600
_cache: dict[str, tuple[float, str]] = {}


def cache_get(key: str) -> str | None:
    entry = _cache.get(key)
    if entry and time.time() - entry[0] < CACHE_TTL:
        return entry[1]
    _cache.pop(key, None)
    return None


def cache_set(key: str, message: str) -> None:
    _cache[key] = (time.time(), message)


class MoodRequest(BaseModel):
    ageGroup: str = Field(..., examples=["20대 중반"])
    positiveRatio: int = 0
    negativeRatio: int = 0
    representativeEmotion: int = Field(..., examples=[75])
    count100: int = 0
    count75: int = 0
    count50: int = 0
    count25: int = 0
    count0: int = 0
    totalCount: int = 0


class MoodResponse(BaseModel):
    message: str


def build_prompt(req: MoodRequest) -> str:
    traits = AGE_TRAITS.get(req.ageGroup, AGE_TRAITS["20대 초반"])
    emotion_guide = EMOTION_GUIDE.get(req.representativeEmotion, EMOTION_GUIDE[50])

    return f"""
오늘 '{req.ageGroup}' 사용자들의 감정 데이터입니다:
- 전체 일기 수: {req.totalCount}개
- 매우 긍정(100): {req.count100}명
- 긍정(75): {req.count75}명
- 보통(50): {req.count50}명
- 부정(25): {req.count25}명
- 매우 부정(0): {req.count0}명
- 긍정 비율: {req.positiveRatio}%
- 부정 비율: {req.negativeRatio}%
- 대표 감정: {req.representativeEmotion}

[연령층 특성]
{traits['특성']}

[말투 가이드]
{traits['말투']}

[공감 포인트]
{traits['공감포인트']}

[메시지 방향]
{emotion_guide}

다음 규칙을 반드시 지켜주세요:
1. 공백 문자 포함 45자 이내로 짧고 다정하게
2. 이모지 1~2개 포함
3. '오늘', '어제', '내일', '금일' 등 특정 날짜 언급 금지
4. 전체 그룹을 아우르는 메시지 (개인이 아닌 연령층 전체에게)
5. 너무 긍정적이거나 가볍지 않게, 진심이 느껴지게
6. 한국어로 작성

JSON 형식으로만 응답:
{{"message": "생성된 메시지"}}
""".strip()


def parse_ai_message(raw_text: str) -> str | None:
    """GPT 응답에서 message 추출"""
    try:
        data = json.loads(raw_text.strip())
        msg = str(data.get("message", "")).strip()
        return msg or None
    except (json.JSONDecodeError, AttributeError):
        return None


@router.get("/")
def health():
    return {"status": "ok", "service": "mood-message-api"}


@router.post("/generate-mood-message", response_model=MoodResponse)
def generate_mood_message(req: MoodRequest):
    cache_key = f"{req.ageGroup}:{req.representativeEmotion}"

    
    cached = cache_get(cache_key)
    if cached:
        logger.info(f"cache hit: {cache_key}")
        return MoodResponse(message=cached)

    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 청춘 일기 앱 '청춘잇다'의 따뜻한 AI 멘토입니다. "
                               "반드시 JSON 객체로만 응답하세요.",
                },
                {"role": "user", "content": build_prompt(req)},
            ],
            temperature=0.9,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        message = parse_ai_message(completion.choices[0].message.content)
    except Exception as e:
        logger.error(f"openai error: {e}")
        raise HTTPException(status_code=502, detail="AI 메시지 생성에 실패했습니다.")

    if not message:
        logger.warning("parse failed")
        raise HTTPException(status_code=502, detail="AI 메시지 생성에 실패했습니다.")

    cache_set(cache_key, message)
    logger.info(f"generated: {cache_key} -> {message}")
    return MoodResponse(message=message)
