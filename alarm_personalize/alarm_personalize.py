"""
청춘잇다 - 일기 회상 알람 (Personalized Recall Alarm)
=====================================================
유저가 일기에 쓴 말을 기억했다가, 그에 맞는 시점에 개인화된 알람을 보내는 기능.

이 모듈 담당 (= AI 파이프라인, 이보현):
  1) POST /recall-alarm/extract  : 일기 → 알람 트리거(JSON) 추출  [일기 저장 시 1회]
  2) POST /recall-alarm/compose  : 트리거 → 알람 문구 생성          [알람 발송 직전]

Java 백엔드 담당: alarm_triggers 테이블 저장 / 스케줄러로 오늘 발동 트리거 SELECT / FCM 발송

※ 팀 컨벤션에 맞춤:
   - 키: GPT_API_KEY (.env), load_dotenv()
   - response_format: json_schema strict (alarm_ment.py와 동일 방식)
   - prefix: /recall-alarm  (alarm_ment의 /alarm 과 충돌 회피)
"""

import os
from datetime import date
from typing import Literal

from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

GPT_API_KEY = os.getenv("GPT_API_KEY")
if not GPT_API_KEY:
    raise ValueError("❌ GPT_API_KEY가 없습니다. .env 파일을 확인해 주세요.")

client = OpenAI(api_key=GPT_API_KEY)
MODEL = "gpt-4o-mini"

# alarm_ment.py의 나이대 분류와 톤을 맞춰서 통일
AGE_TONE = {
    "고등학생": "친구처럼 편하게, 이모지 조금, 짧고 가볍게",
    "20대 초반": "다정한 또래 친구처럼, 공감 위주, 담백하게",
    "20대 중반": "같이 고생하는 또래처럼, 공감하고 응원하듯",
    "20대 후반~30대 초반": "차분하고 존중하는 어른 친구처럼, 진중하게",
    "30대 중반 이상": "따뜻하고 예의 있게, 잔소리 아닌 안부처럼",
}


def categorize_age(age: int) -> str:
    """alarm_ment.py와 동일한 나이 분류 기준."""
    if age <= 19:
        return "고등학생"
    elif 20 <= age <= 23:
        return "20대 초반"
    elif 24 <= age <= 26:
        return "20대 중반"
    elif 27 <= age <= 33:
        return "20대 후반~30대 초반"
    else:
        return "30대 중반 이상"


router = APIRouter(prefix="/recall-alarm", tags=["RecallAlarm"])


@router.get("/")
def health_check():
    return {"message": "Recall Alarm AI is running!"}


# =============================================================
# ① 추출부  POST /recall-alarm/extract
#    일기 저장할 때 호출. 미래 이벤트 / 감정 패턴을 뽑아낸다.
# =============================================================

class ExtractRequest(BaseModel):
    user_id: str
    diary_text: str
    written_date: date          # 일기 쓴 날짜 (상대날짜 계산 기준!)
    age: int                    # 나이 (categorize_age로 그룹핑)


EXTRACT_SYSTEM_PROMPT = """너는 일기에서 '나중에 알람으로 챙겨줄 거리'를 뽑아내는 추출기다.

기준 날짜(오늘): {today} (요일: {weekday})
상대 표현("내일", "다음 주 수요일", "3일 뒤")은 이 기준 날짜로 계산해 YYYY-MM-DD로 변환한다.

3종류의 트리거를 찾아 triggers 배열에 담는다. 없으면 빈 배열.

1) event  : 날짜 있는 미래 일 (발표, 시험, 면접, 병원, 생일, 여행 등)
2) pattern: "X하면 기분이 좋아진다/나빠진다" 감정 규칙
3) care   : 지금 일기가 많이 힘들어 보임 (누적 판단은 백엔드가 함)

[시각 처리 규칙]
- 일기에 구체적 시각이 있으면 alarm_time을 "HH:mm"(24시간제)로 변환한다.
  예: "오후 3시" → "15:00", "아침 8시반" → "08:30", "저녁 7시" → "19:00", "밤 11시" → "23:00"
- 구체적 시각이 없으면 alarm_time을 대략적 시간대로:
  아침 느낌 → "morning", 낮 → "afternoon", 저녁/밤 → "evening"
- 시간 정보가 아예 없으면 event는 "morning", care는 "evening", pattern은 null.

확실하지 않으면 넣지 마라(환각 금지). 진단/병명 금지.
"""

# alarm_ment.py처럼 json_schema strict 방식 사용 → 구조 100% 보장
EXTRACT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "alarm_triggers",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "triggers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["event", "pattern", "care"]},
                            "target_date": {"type": ["string", "null"]},
                            "topic": {"type": ["string", "null"]},
                            "emotion": {"type": ["string", "null"]},
                            "trigger_emotion": {"type": ["string", "null"]},
                            "action": {"type": ["string", "null"]},
                            "effect": {"type": ["string", "null"], "enum": ["positive", "negative", None]},
                            "reason": {"type": ["string", "null"]},
                            "alarm_time": {"type": ["string", "null"], "description": "구체적 시각이면 HH:mm(예 15:00), 없으면 morning/afternoon/evening, 아예 없으면 null"},
                        },
                        "required": ["type", "target_date", "topic", "emotion",
                                     "trigger_emotion", "action", "effect", "reason", "alarm_time"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["triggers"],
            "additionalProperties": False,
        },
    },
}


@router.post("/extract")
def extract_triggers(req: ExtractRequest):
    import json
    weekday_kr = ["월", "화", "수", "목", "금", "토", "일"][req.written_date.weekday()]
    system = EXTRACT_SYSTEM_PROMPT.format(today=req.written_date.isoformat(), weekday=weekday_kr)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            response_format=EXTRACT_SCHEMA,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"[일기]\n{req.diary_text}"},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        triggers = data.get("triggers", [])
    except Exception as e:
        # 추출 실패해도 일기 저장은 막으면 안 됨 → 빈 배열 폴백
        print(f"❌ [extract] 실패, 빈 배열로 폴백: {e}")
        triggers = []

    return {"user_id": req.user_id, "age_group": categorize_age(req.age), "triggers": triggers}


# =============================================================
# ② 생성부  POST /recall-alarm/compose
#    스케줄러가 오늘 발동할 트리거를 넘겨주면 알람 문구를 만든다.
# =============================================================

class ComposeRequest(BaseModel):
    trigger: dict               # extract가 뱉은 트리거 하나 그대로
    age: int
    diary_excerpt: str | None = None   # (선택) 그때 일기 한 줄 → 인용하면 생생함


COMPOSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "alarm_message",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["title", "body"],
            "additionalProperties": False,
        },
    },
}


def _build_context(t: dict, excerpt: str | None) -> str:
    ttype = t.get("type")
    if ttype == "event":
        base = f"오늘은 유저가 '{t.get('topic')}'(을)를 하는 날. 그때 '{t.get('emotion')}' 감정이었음."
    elif ttype == "pattern":
        rel = "기분이 좋아진다고" if t.get("effect") == "positive" else "기분이 나빠진다고"
        base = (f"유저는 '{t.get('action')}'을 하면 {rel} 했음. "
                f"지금 '{t.get('trigger_emotion')}' 상태라 그 행동을 부드럽게 권하는 상황.")
    else:  # care
        base = f"유저가 요즘 힘들어 보임 ({t.get('reason')}). 조심스럽게 안부만 묻는 상황."

    if excerpt:
        base += f'\n(참고 - 그때 일기 한 줄: "{excerpt}")'
    return base


@router.post("/compose")
def compose_alarm(req: ComposeRequest):
    import json
    age_group = categorize_age(req.age)
    tone = AGE_TONE.get(age_group, AGE_TONE["20대 초반"])
    context = _build_context(req.trigger, req.diary_excerpt)

    system = (
        f"너는 청춘잇다 앱의 다정한 알람 문구 작가다. "
        f"유저가 예전에 일기에 쓴 내용을 기억했다가 말을 거는 톤이다.\n"
        f"말투: {tone}\n"
        f"규칙: 2~3문장으로 짧게. '엿들은' 느낌 금지 - 유저가 일기에 써준 정보임을 자연스럽게. "
        f"진단/훈계/명령 금지. 곁에 있는 친구 느낌."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.8,
            response_format=COMPOSE_SCHEMA,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": context},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        title, body = data["title"], data["body"]
    except Exception as e:
        print(f"❌ [compose] 실패, 기본 문구로 폴백: {e}")
        title, body = "오늘의 안부", "잠깐, 오늘 하루는 어땠어? 일기로 남겨볼까?"

    alarm_time = req.trigger.get("alarm_time") or "evening"
    return {"title": title, "body": body, "alarm_time": alarm_time}
