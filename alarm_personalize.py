from datetime import date, datetime, timedelta
from enum import Enum
from typing import Literal

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["GPT_API_KEY"])
router = APIRouter(prefix="/alarm", tags=["alarm"])

MODEL = "gpt-4o-mini"


# ─────────────────────────────────────────────────────────────
# 공통: 나이대별 말투 (다른 엔드포인트랑 톤 통일)
# ─────────────────────────────────────────────────────────────
AGE_TONE = {
    "10s": "친구처럼 편하게, 이모지 조금, 짧고 가볍게",
    "20s": "다정한 또래 친구처럼, 공감 위주, 담백하게",
    "30s": "차분하고 존중하는 어른 친구처럼, 담백하고 진중하게",
    "40s+": "따뜻하고 예의 있게, 잔소리 아닌 안부처럼",
}


# =============================================================
# ① 추출부  POST /alarm/extract
#    일기 저장할 때 호출. 미래 이벤트 / 감정 패턴을 뽑아낸다.
# =============================================================

class ExtractRequest(BaseModel):
    user_id: str
    diary_text: str
    written_date: date          # 일기 쓴 날짜 (상대날짜 계산 기준!)
    emotion: str                # 프론트/감정분석에서 넘어온 오늘 감정
    age_group: Literal["10s", "20s", "30s", "40s+"] = "20s"


# 추출 프롬프트 = 이 기능의 심장.
# 핵심 3가지: (1) 상대날짜를 실제 날짜로 환산  (2) 없으면 빈 배열
#            (3) 반드시 JSON만 출력
EXTRACT_SYSTEM_PROMPT = """너는 일기에서 '나중에 알람으로 챙겨줄 거리'를 뽑아내는 추출기다.
오직 JSON만 출력한다. 설명/인사/코드블록 금지.

기준 날짜(오늘): {today}  (요일: {weekday})
상대 표현("내일", "다음 주 수요일", "3일 뒤")은 이 기준 날짜로 계산해 YYYY-MM-DD로 변환한다.

다음 3종류의 트리거를 찾아 "triggers" 배열에 담는다. 없으면 빈 배열 [].

1) event  : 날짜가 있는 미래 일/약속 (발표, 시험, 면접, 병원, 생일, 여행 등)
   { "type":"event", "target_date":"YYYY-MM-DD", "topic":"발표",
     "emotion":"불안", "alarm_time":"morning" }
   - alarm_time: 그날 언제 알려주면 좋을지 → "morning"/"afternoon"/"evening" 중 택1

2) pattern: "X하면 기분이 좋아진다/나빠진다" 같은 감정 규칙
   { "type":"pattern", "trigger_emotion":"무기력", "action":"운동",
     "effect":"positive" }
   - effect: 그 행동이 기분에 positive / negative 인지

3) care   : 지금 이 일기 자체가 많이 힘들어 보이면 (누적 판단은 백엔드가 함)
   { "type":"care", "reason":"수면 문제 반복 호소", "alarm_time":"evening" }

규칙:
- 확실하지 않으면 넣지 마라. 억지로 만들지 마라 (환각 금지).
- 하나도 없으면 {"triggers": []}.
- 진단하거나 병명 붙이지 마라.
"""


class Trigger(BaseModel):
    type: Literal["event", "pattern", "care"]
    # event
    target_date: date | None = None
    topic: str | None = None
    emotion: str | None = None
    alarm_time: Literal["morning", "afternoon", "evening"] | None = None
    # pattern
    trigger_emotion: str | None = None
    action: str | None = None
    effect: Literal["positive", "negative"] | None = None
    # care
    reason: str | None = None


class ExtractResponse(BaseModel):
    user_id: str
    triggers: list[Trigger]


@router.post("/extract", response_model=ExtractResponse)
def extract_triggers(req: ExtractRequest):
    weekday_kr = ["월", "화", "수", "목", "금", "토", "일"][req.written_date.weekday()]
    system = EXTRACT_SYSTEM_PROMPT.format(
        today=req.written_date.isoformat(), weekday=weekday_kr
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,  # 추출은 창의성 낮게 = 안정적으로
            response_format={"type": "json_object"},  # JSON 강제
            messages=[
                {"role": "system", "content": system},
                {"role": "user",
                 "content": f"[일기]\n{req.diary_text}\n\n[오늘 감정] {req.emotion}"},
            ],
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        triggers = [Trigger(**t) for t in data.get("triggers", [])]
    except Exception as e:
        # 추출 실패해도 일기 저장 자체는 막으면 안 됨 → 빈 배열로 폴백
        print(f"[extract] fallback: {e}")
        triggers = []

    return ExtractResponse(user_id=req.user_id, triggers=triggers)


# =============================================================
# ② 생성부  POST /alarm/compose
#    스케줄러가 "오늘 발동할 트리거"를 찾아서 넘겨주면,
#    나이대 톤을 입힌 실제 알람 문구를 만든다.
# =============================================================

class ComposeRequest(BaseModel):
    trigger: Trigger
    age_group: Literal["10s", "20s", "30s", "40s+"] = "20s"
    # (선택) 원본 일기 한 줄을 같이 주면 "그때 네가 이랬잖아" 인용이 더 생생해짐
    diary_excerpt: str | None = None


class ComposeResponse(BaseModel):
    title: str      # 푸시 제목
    body: str       # 푸시 본문
    alarm_time: str  # morning/afternoon/evening (백엔드가 실제 시각으로 변환)


COMPOSE_SYSTEM_PROMPT = """너는 청춘잇다 앱의 다정한 알람 문구 작가다.
유저가 예전에 일기에 쓴 내용을 기억했다가 말을 거는 톤이다.

말투: {tone}

규칙:
- 2~3문장, 짧게. 푸시 알림이니까.
- "엿들은" 느낌 절대 금지. "네가 말해줬잖아 / 일기에 썼잖아" 처럼 유저가 준 정보임을 자연스럽게.
- 진단/훈계/명령 금지. 곁에 있는 친구 느낌.
- 반드시 JSON만: {{"title": "...", "body": "..."}}
"""


def _build_context(t: Trigger, excerpt: str | None) -> str:
    """트리거 종류별로 GPT에게 줄 상황 설명을 만든다."""
    if t.type == "event":
        base = f"오늘은 유저가 '{t.topic}'(을)를 하는 날. 그때 '{t.emotion}' 감정이었음."
    elif t.type == "pattern":
        rel = "기분이 좋아진다고" if t.effect == "positive" else "기분이 나빠진다고"
        base = (f"유저는 '{t.action}'을 하면 {rel} 했음. "
                f"지금 '{t.trigger_emotion}' 상태라 그 행동을 부드럽게 권하는 상황.")
    else:  # care
        base = f"유저가 요즘 힘들어 보임 ({t.reason}). 조심스럽게 안부만 묻는 상황."

    if excerpt:
        base += f"\n(참고 - 그때 일기 한 줄: \"{excerpt}\")"
    return base


@router.post("/compose", response_model=ComposeResponse)
def compose_alarm(req: ComposeRequest):
    tone = AGE_TONE.get(req.age_group, AGE_TONE["20s"])
    system = COMPOSE_SYSTEM_PROMPT.format(tone=tone)
    context = _build_context(req.trigger, req.diary_excerpt)

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.8,  # 문구는 살짝 창의적으로
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": context},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        title = data.get("title", "청춘잇다")
        body = data["body"]
    except Exception as e:
        # 생성 실패 시 안전한 기본 문구 (알람이 아예 안 나가는 것보단 나음)
        print(f"[compose] fallback: {e}")
        title, body = "오늘의 안부", "잠깐, 오늘 하루는 어땠어? 일기로 남겨볼까?"

    alarm_time = req.trigger.alarm_time or "evening"
    return ComposeResponse(title=title, body=body, alarm_time=alarm_time)
