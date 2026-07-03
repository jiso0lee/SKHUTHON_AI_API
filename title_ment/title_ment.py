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
from fastapi import FastAPI
from pydantic import BaseModel, Field
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mood-message")

# ─────────────────────────────
# OpenAI 설정
# ─────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = "gpt-4o-mini"  # 빠르고 저렴 → 5초 타임아웃 안에 안정적

client = OpenAI(
    api_key=OPENAI_API_KEY or "not-set",  # 키 없어도 서버는 기동 → 호출 시 폴백 처리
    timeout=4.0,       # 백엔드 타임아웃(5초)보다 짧게
    max_retries=0,     # 재시도하면 5초 초과하므로 즉시 폴백
)

app = FastAPI(title="청춘잇다 Mood Message API")

# ─────────────────────────────
# 연령층별 프롬프트 가이드
# ─────────────────────────────
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

# ─────────────────────────────
# 폴백 메시지 (AI 장애 시)
# ─────────────────────────────
FALLBACK_MESSAGES = {
    100: "빛나는 마음들이 모인 하루였어요, 이 따뜻한 에너지 오래오래 간직하길 바라요 ✨",
    75: "잘 지내고 있는 마음들이 느껴져요, 스스로를 충분히 칭찬해줘도 좋아요 🌿",
    50: "평범해 보이는 순간들도 모두 소중한 시간이에요, 가끔은 쉬어가도 괜찮아요 ☁️",
    25: "마음이 무거운 분들이 많았네요, 그 감정도 충분히 소중하니 스스로를 다독여주세요 🍀",
    0: "많이 힘들었을 마음들에게, 잠시 멈춰 쉬어도 괜찮다고 말해주고 싶어요 🌙",
}

# ─────────────────────────────
# 캐시 (ageGroup + representativeEmotion 기준, TTL 1시간)
# ─────────────────────────────
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


# ─────────────────────────────
# Request / Response 모델
# ─────────────────────────────
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


# ─────────────────────────────
# 프롬프트 생성
# ─────────────────────────────
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
1. 2문장 이내로 짧고 다정하게
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


def get_fallback(emotion: int) -> str:
    return FALLBACK_MESSAGES.get(emotion, FALLBACK_MESSAGES[50])


# ─────────────────────────────
# 엔드포인트
# ─────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "mood-message-api"}


@app.post("/generate-mood-message", response_model=MoodResponse)
def generate_mood_message(req: MoodRequest):
    cache_key = f"{req.ageGroup}:{req.representativeEmotion}"

    # 1) 캐시 히트 시 즉시 반환 (백엔드 5초 타임아웃 대비)
    cached = cache_get(cache_key)
    if cached:
        logger.info(f"cache hit: {cache_key}")
        return MoodResponse(message=cached)

    # 2) GPT 호출
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
            response_format={"type": "json_object"},  # JSON 강제
        )
        message = parse_ai_message(completion.choices[0].message.content)

        if message:
            cache_set(cache_key, message)
            logger.info(f"generated: {cache_key} -> {message}")
            return MoodResponse(message=message)

        logger.warning("parse failed, falling back")
    except Exception as e:
        logger.error(f"openai error: {e}")

    # 3) 실패 시 폴백 (AI 서버 자체 폴백 → 백엔드 폴백은 2차 방어)
    return MoodResponse(message=get_fallback(req.representativeEmotion))
