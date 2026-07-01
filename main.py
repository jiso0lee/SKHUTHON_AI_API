import os
import json
import time
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from fastapi import FastAPI

# ==========================================
# 0. 보안: .env 파일에서 환경 변수 불러오기
# ==========================================
load_dotenv()

# ==========================================
# 1. 환경 설정 및 API 엔드포인트 세팅
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BACKEND_GET_DIARIES_URL = os.getenv("BACKEND_GET_URL")
BACKEND_POST_COMMENTS_URL = os.getenv("BACKEND_POST_URL")

# API 키 누락 방어막
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY가 없습니다. .env 파일을 확인해 주세요.")

client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# 2. 연령대별 특성 가이드 사전 (Prompt 주입용)
# ==========================================
AGE_TRAITS = {
    "고등학생": "학업 스트레스, 진로에 대한 고민, 친구 관계, 풋풋한 일상",
    "20대 초반": "대학 생활, 아르바이트, 새로운 만남, 진로 탐색, 자유로움",
    "20대 중반": "취업 준비의 압박, 사회 초년생의 낯섦, 연애, 독립",
    "20대 후반~30대 초반": "직장 생활의 고충, 커리어 발전 고민, 결혼 및 미래에 대한 불안감",
    "30대 중반 이상": "삶의 안정과 번아웃, 가족/육아, 건강, 인생의 방향성 재점검"
}

# ==========================================
# 3. 연령대 분류 함수 (Rule-based)
# ==========================================
def categorize_age(age):
    """나이를 입력받아 5가지 그룹으로 분류합니다."""
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

# ==========================================
# 4. 메인 파이프라인 실행 함수
# ==========================================
def run_daily_ai_analysis():
    print("🚀 일일 AI 맞춤형 분석 및 멘트 매핑 작업을 시작합니다...")
    
    # [STEP 1] 백엔드에서 오늘 쌓인 일기 데이터 받아오기
    try:
        response = requests.get(BACKEND_GET_DIARIES_URL)
        response.raise_for_status()
        diaries_data = response.json().get("data", [])
    except Exception as e:
        print(f"❌ 데이터 수신 실패: {e}")
        return

    if not diaries_data:
        print("📭 오늘 분석할 일기 데이터가 없습니다.")
        return

    # [STEP 2] 딕셔너리 반복문으로 안전하게 전처리 및 그룹화 (KeyError 방어)
    grouped_data = {} 
    
    for diary in diaries_data:
        # 1. 비공개 일기 제외 (2중 안전장치)
        if diary.get('isPublic') == False:
            continue
            
        # 2. 나이 정보 안전하게 가져오기 (diary['age'] 에러 대비)
        age = diary.get('age')
        
        # 나이 데이터가 누락되었을 경우 서버가 다운되는 것을 막고 패스하기
        if age is None:
            diary_id = diary.get('id', '알수없음')
            print(f"⚠️ 경고: 일기 ID {diary_id}에 'age' 데이터가 없어 분석에서 제외합니다.")
            continue 
            
        # 나이를 바탕으로 연령 그룹 이름 찾기
        age_group = categorize_age(age)
        
        # 3. 연령대별 그룹에 담기
        if age_group not in grouped_data:
            grouped_data[age_group] = []
            
        grouped_data[age_group].append(diary)

    # 최종적으로 백엔드에 한 번에 보낼 리스트 준비
    final_payload = []

    # [STEP 3] 연령대별로 그룹을 순회하며 AI 분석 진행
    for age_group, diaries_list in grouped_data.items():
        print(f"🧠 [{age_group}] 감정 분류 및 맞춤형 멘트 생성 중...")
        
        # AI 프롬프트 크기를 줄이기 위해 id와 content만 추려내기
        diaries_for_prompt = [
            {"id": d.get("id"), "content": d.get("content")} 
            for d in diaries_list 
            if d.get("id") is not None and d.get("content") is not None
        ]
        
        # 만약 일기가 없다면 패스
        if not diaries_for_prompt:
            continue
            
        # 현재 연령대에 딱 맞는 특성 키워드 가져오기
        current_group_traits = AGE_TRAITS.get(age_group, "다양한 일상과 고민")
        
        # 프롬프트 작성
        prompt = f"""
        당신은 따뜻하고 공감 능력이 뛰어난 다이어리 앱의 AI 멘토입니다.
        아래는 오늘 '{age_group}' 사용자들이 작성한 일기(ID와 내용) 모음입니다.

        [연령대별 특성 가이드: {age_group}]
        - 핵심 키워드: {current_group_traits}

        [일기 데이터]
        {json.dumps(diaries_for_prompt, ensure_ascii=False)}

        다음 두 가지 작업을 수행해 주세요:
        1. 감정 판별: 각 일기(id)의 텍스트에서 추출한 감정을 'positive'(긍정) 또는 'negative'(부정) 두 케이스 중 하나로만 판별하세요.
        2. 추천 멘트 작성: 위에서 제시한 '{age_group}'의 특성과 일기 내용들의 전반적인 분위기를 자연스럽게 연결하여, 이 그룹 전체를 아우르는 추천 멘트를 작성하세요.
           - positive_comment: 일기에서 느껴지는 활기찬 에너지나 성취감을 더욱 북돋아 주고, 연령대에 맞게 진심으로 응원하고 칭찬하는 멘트.
           - negative_comment: 현재 연령대에서 겪을 수 있는 힘듦과 고민을 다독이고, 따뜻한 위로와 가벼운 휴식을 제안하는 멘트.
           - 분량 제한: 2~3문장으로 짧고 다정하게 작성할 것. (예시: "취업 준비로 마음이 많이 무거우셨군요. 좋아하는 음악을 들으며 스스로를 꼭 안아주는 시간을 가져보세요.")
           - 🚨[주의사항]: 메일로 늦게 발송되는 점을 고려하여 멘트 작성 시 '어제', '오늘', '내일', '금일' 등 특정 시점이나 날짜를 지칭하는 단어는 절대 사용하지 말 것.

        반드시 지정된 JSON 형식으로만 응답해 주세요.
        """

        try:
            # Gemini 2.0 Flash 모델 호출 (JSON 구조화 응답 기능 활용)
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "classifications": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "integer"},
                                        "sentiment": {"type": "string", "enum": ["positive", "negative"]}
                                    }
                                }
                            },
                            "positive_comment": {"type": "string"},
                            "negative_comment": {"type": "string"}
                        },
                        "required": ["classifications", "positive_comment", "negative_comment"]
                    }
                )
            )
            
            # AI 결과 해석하기
            ai_result = json.loads(response.text)
            pos_comment = ai_result["positive_comment"]
            neg_comment = ai_result["negative_comment"]
            
            # [STEP 4] AI가 분류한 일기별 감정에 따라, 멘트를 1:1로 매칭
            for item in ai_result["classifications"]:
                diary_id = item["id"]
                sentiment = item["sentiment"]
                
                # 감정이 긍정이면 응원 멘트, 부정이면 위로 멘트 매칭
                matched_comment = pos_comment if sentiment == "positive" else neg_comment
                
                # 최종 데이터 딕셔너리 구성
                final_payload.append({
                    "id": diary_id, 
                    "aiComment": matched_comment
                })
            
        except Exception as e:
            print(f"❌ [{age_group}] AI 분석 중 오류 발생: {e}")
            continue

        # API 호출이 하나 끝날 때마다 15초씩 대기
        print("⏳ API 속도 제한 방지를 위해 15초 대기 중...")
        time.sleep(15)

    # [STEP 5] 모든 데이터를 백엔드 서버에 단 한 번의 요청으로 일괄 전송 (Batch)
    if final_payload:
        try:
            post_response = requests.post(
                BACKEND_POST_COMMENTS_URL, 
                json={"recommendations": final_payload}
            )
            post_response.raise_for_status()
            print(f"✅ 총 {len(final_payload)}개의 일기에 맞춤형 AI 추천 멘트 매핑 완료 및 백엔드 전송 성공!")
        except Exception as e:
            print(f"❌ 백엔드 전송 최종 실패: {e}")

# FastAPI 웹 서버 세팅
app = FastAPI()

# 1. Render가 서버가 살아있는지 확인하기 위한 기본 주소 (Health Check)
@app.get("/")
def health_check():
    return {"message": "AI Server is running perfectly!"}

# 2. 백엔드에서 특정 주소로 요청을 보내면 AI 분석 코드가 실행되도록 연결
@app.get("/run-ai")
def trigger_ai_analysis():
    run_daily_ai_analysis()
    return {"message": "AI analysis triggered and completed."}