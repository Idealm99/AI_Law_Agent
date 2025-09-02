# model/prompts/prompt.py

from textwrap import dedent
from langchain_core.prompts import ChatPromptTemplate

# --- 정보 추출 및 평가 프롬프트 ---
def get_extract_prompt(law_name: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", f"""당신은 {law_name} 전문가입니다. 주어진 문서에서 질문과 관련된 주요 사실과 정보를 3~5개 정도 추출하세요.
        각 추출된 정보에 대해 다음 두 가지 측면을 0에서 1 사이의 점수로 평가하세요:
        1. 질문과의 관련성
        2. 답변의 충실성 (질문에 대한 완전하고 정확한 답변을 제공할 수 있는 정도)

        추출 형식:
        1. [추출된 정보]
        - 관련성 점수: [0-1 사이의 점수]
        - 충실성 점수: [0-1 사이의 점수]
        ...

        마지막으로, 추출된 정보를 종합하여 질문에 대한 전반적인 답변 가능성을 0에서 1 사이의 점수로 평가하세요."""),
        ("human", "[질문]\\n{question}\\n\\n[문서 내용]\\n{document_content}")
    ])

# --- 질문 재작성 프롬프트 ---
def get_rewrite_prompt(law_name: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", f"""당신은 {law_name} 전문가입니다. 주어진 원래 질문과 추출된 정보를 바탕으로, 더 관련성 있고 충실한 정보를 찾기 위해 검색 쿼리를 개선해주세요.
        다음 사항을 고려하여 검색 쿼리를 개선하세요:
        1. 원래 질문의 핵심 요소
        2. 추출된 정보의 관련성 점수 및 충실성 점수
        3. 부족한 정보나 더 자세히 알아야 할 부분

        개선된 검색 쿼리 작성 단계:
        1. 2-3개의 검색 쿼리를 제안하세요.
        2. 각 쿼리는 구체적이고 간결해야 합니다.
        3. {law_name}과 관련된 전문 용어를 적절히 활용하세요.
        
        마지막으로, 제안된 쿼리 중 가장 효과적일 것 같은 쿼리를 선택하고 그 이유를 설명하세요."""),
        ("human", "원래 질문: {question}\\n\\n추출된 정보:\\n{extracted_info}\\n\\n위 지침에 따라 개선된 검색 쿼리를 작성해주세요.")
    ])

# --- 노드 답변 생성 프롬프트 ---
def get_answer_prompt(law_name: str, source_example: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", f"""당신은 {law_name} 전문가입니다. 주어진 질문과 추출된 정보를 바탕으로 답변을 생성해주세요.
        답변은 마크다운 형식으로 작성하며, 각 정보의 출처를 명확히 표시해야 합니다.
        답변 구조:
        1. 질문에 대한 직접적인 답변
        2. 관련 법률 조항 및 해석 (또는 관련 출처 및 링크)
        3. 추가 설명 또는 예시 (필요한 경우)
        4. 결론 및 요약
        각 섹션에서 사용된 정보의 출처를 괄호 안에 명시하세요. 예: ({source_example})"""),
        ("human", "질문: {question}\\n\\n추출된 정보:\\n{extracted_info}\\n\\n위 지침에 따라 최종 답변을 작성해주세요.")
    ])

# --- 질문 라우팅 프롬프트 ---
route_system_prompt = dedent("""You are an AI assistant specializing in routing user questions to the appropriate tools.
Use the following guidelines:
- For questions specifically about legal provisions or articles of the privacy protection law (개인정보 보호법), use the search_personal tool.
- For questions specifically about legal provisions or articles of the labor law (근로기준법), use the search_labor tool.
- For questions specifically about legal provisions or articles of the housing law (주택임대차보호법), use the search_housing tool.
- For any other information, including questions related to these laws but not directly about specific legal provisions, or for the most up-to-date data, use the search_web tool.
Always choose all of the appropriate tools based on the user's question. 
If a question is about a law but doesn't seem to be asking about specific legal provisions, include both the relevant law search tool and the search_web tool.""")

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", route_system_prompt),
        ("human", "{question}"),
    ]
)

# --- 최종 답변 생성 프롬프트 ---
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant answering questions based on provided documents. Follow these guidelines:
1. Use only information from the given documents.
2. If the document lacks relevant info, say "제공된 정보로는 충분한 답변을 할 수 없습니다."
3. Cite the source of information for each sentence in your answer.
4. Keep answers concise and clear."""),
    ("human", "Answer the following question using these documents:\\n\\n[Documents]\\n{documents}\\n\\n[Question]\\n{question}"),
])

# --- LLM Fallback 프롬프트 ---
fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant. Provide a helpful answer to the user's question."),
    ("human", "{question}"),
])

# --- 답변 평가 프롬프트 ---
evaluation_prompt = dedent("""
당신은 AI 어시스턴트가 생성한 답변을 평가하는 전문가입니다. 주어진 질문과 답변을 평가하고, 60점 만점으로 점수를 매기세요. 다음 기준을 사용하여 평가하십시오:
1. 정확성 (10점)
2. 관련성 (10점)
3. 완전성 (10점)
4. 인용 정확성 (10점)
5. 명확성과 간결성 (10점)
6. 객관성 (10점)
평가 과정:
1. 주어진 질문과 답변을 주의 깊게 읽으십시오.
2. 필요한 경우, 다음 도구를 사용하여 추가 정보를 수집하세요:
   - web_search, personal_law_search, labor_law_search, housing_law_search
3. 각 기준에 대해 1-10점 사이의 점수를 매기세요.
4. 총점을 계산하세요 (60점 만점).

출력 형식:
{
  "scores": {
    "accuracy": 0, "relevance": 0, "completeness": 0, "citation_accuracy": 0, "clarity_conciseness": 0, "objectivity": 0
  },
  "total_score": 0,
  "brief_evaluation": "간단한 평가 설명"
}

최종 출력에는 각 기준의 점수, 총점, 그리고 간단한 평가 설명만 포함하세요.
""")