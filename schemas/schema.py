# schemas/schema.py

from typing import List, TypedDict, Annotated, Optional
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# --- Corrective RAG 에이전트 상태 정의 ---

class CorrectiveRagState(TypedDict):
    """Corrective RAG의 기본 상태"""
    question: str
    generation: str
    documents: List[Document]
    num_generations: int

class InformationStrip(BaseModel):
    """추출된 정보 조각의 내용, 출처, 관련성 점수"""
    content: str = Field(..., description="추출된 정보 내용")
    source: str = Field(..., description="정보의 출처(법률 조항 또는 URL 등)")
    relevance_score: float = Field(..., ge=0, le=1, description="질의에 대한 관련성 점수 (0에서 1 사이)")
    faithfulness_score: float = Field(..., ge=0, le=1, description="답변의 충실성 점수 (0에서 1 사이)")

class ExtractedInformation(BaseModel):
    """추출된 정보 조각들과 전반적인 답변 가능성 점수"""
    strips: List[InformationStrip] = Field(..., description="추출된 정보 조각들")
    query_relevance: float = Field(..., ge=0, le=1, description="질의에 대한 전반적인 답변 가능성 점수 (0에서 1 사이)")

class RefinedQuestion(BaseModel):
    """개선된 질문과 그 이유"""
    question_refined: str = Field(..., description="개선된 질문")
    reason: str = Field(..., description="이유")

# --- 각 법률 및 웹 검색 에이전트 상태 정의 ---

class PersonalRagState(CorrectiveRagState):
    """개인정보보호법 RAG 에이전트 상태"""
    rewritten_query: str
    extracted_info: Optional[List[InformationStrip]]
    node_answer: Optional[str]

class LaborRagState(CorrectiveRagState):
    """근로기준법 RAG 에이전트 상태"""
    rewritten_query: str
    extracted_info: Optional[List[InformationStrip]]
    node_answer: Optional[str]

class HousingRagState(CorrectiveRagState):
    """주택임대차보호법 RAG 에이전트 상태"""
    rewritten_query: str
    extracted_info: Optional[List[InformationStrip]]
    node_answer: Optional[str]

class SearchRagState(CorrectiveRagState):
    """웹 검색 RAG 에이전트 상태"""
    rewritten_query: str
    extracted_info: Optional[List[InformationStrip]]
    node_answer: Optional[str]

# --- 메인 그래프(Supervisor) 상태 정의 ---

class ResearchAgentState(TypedDict):
    """메인 에이전트의 상태"""
    question: str
    answers: Annotated[List[str], lambda x, y: x + y]
    final_answer: str
    datasources: List[str]
    evaluation_report: Optional[dict]
    user_decision: Optional[str]

# --- 라우팅을 위한 데이터 모델 ---

class ToolSelector(BaseModel):
    """사용자 질문에 가장 적합한 도구를 선택합니다."""
    tool: str = Field(
        description="사용자 질문에 따라 도구 중 하나를 선택합니다.",
        enum=["search_personal", "search_labor", "search_housing", "search_web"]
    )

class ToolSelectors(BaseModel):
    """사용자 질문에 적합한 도구들을 선택합니다."""
    tools: List[ToolSelector] = Field(
        description="사용자 질문에 따라 하나 이상의 도구를 선택합니다.",
    )