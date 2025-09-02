# model/graph/langgraph.py

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from schemas.schema import (
    ResearchAgentState, PersonalRagState, LaborRagState, HousingRagState, SearchRagState
)
from model.graph.nodes import create_rag_nodes, analyze_question_tool_search, route_datasources_tool_search, \
    answer_final, llm_fallback, evaluate_answer_node, human_review_node
from model.tools.langchain_tools import personal_law_search, labor_law_search, housing_law_search, web_search

# --- Corrective RAG 에이전트 생성 함수 ---
def create_rag_agent(law_name: str, search_tool: callable, state_type: type):
    retrieve, extract_evaluate, rewrite, generate, should_continue = create_rag_nodes(law_name, search_tool, state_type)
    
    workflow = StateGraph(state_type)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("extract_and_evaluate", extract_evaluate)
    workflow.add_node("rewrite_query", rewrite)
    workflow.add_node("generate_answer", generate)
    
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "extract_and_evaluate")
    workflow.add_conditional_edges(
        "extract_and_evaluate",
        should_continue,
        {"continue": "rewrite_query", "end": "generate_answer"}
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate_answer", END)
    
    return workflow.compile()

# --- 각 RAG 에이전트 컴파일 ---
personal_law_agent = create_rag_agent("개인정보보호법", personal_law_search, PersonalRagState)
labor_law_agent = create_rag_agent("근로기준법", labor_law_search, LaborRagState)
housing_law_agent = create_rag_agent("주택임대차보호법", housing_law_search, HousingRagState)
search_web_agent = create_rag_agent("인터넷 검색", web_search, SearchRagState)

# --- Supervisor 노드 정의 ---
def personal_rag_node_supervisor(state: ResearchAgentState) -> dict:
    question = state["question"]
    result = personal_law_agent.invoke({"question": question})
    return {"answers": [result["node_answer"]]}

def labor_rag_node_supervisor(state: ResearchAgentState) -> dict:
    question = state["question"]
    result = labor_law_agent.invoke({"question": question})
    return {"answers": [result["node_answer"]]}

def housing_rag_node_supervisor(state: ResearchAgentState) -> dict:
    question = state["question"]
    result = housing_law_agent.invoke({"question": question})
    return {"answers": [result["node_answer"]]}

def web_rag_node_supervisor(state: ResearchAgentState) -> dict:
    question = state["question"]
    result = search_web_agent.invoke({"question": question})
    return {"answers": [result["node_answer"]]}

# --- Supervisor 그래프 빌드 ---
nodes = {
    "analyze_question": analyze_question_tool_search,
    "search_personal": personal_rag_node_supervisor,
    "search_labor": labor_rag_node_supervisor,
    "search_housing": housing_rag_node_supervisor,
    "search_web": web_rag_node_supervisor,
    "generate_answer": answer_final,
    "llm_fallback": llm_fallback,
    "evaluate_answer": evaluate_answer_node,
    "human_review": human_review_node,
}

search_builder = StateGraph(ResearchAgentState)

for node_name, node_func in nodes.items():
    search_builder.add_node(node_name, node_func)

search_builder.add_edge(START, "analyze_question")
search_builder.add_conditional_edges(
    "analyze_question",
    route_datasources_tool_search,
    ["search_personal", "search_labor", "search_housing", "search_web", "llm_fallback"]
)

for node in ["search_personal", "search_labor", "search_housing", "search_web"]:
    search_builder.add_edge(node, "generate_answer")

search_builder.add_edge("generate_answer", "evaluate_answer")
search_builder.add_edge("evaluate_answer", "human_review")
search_builder.add_conditional_edges(
    "human_review",
    lambda x: "approved" if x.get("user_decision") == "approved" else "rejected",
    {
        "approved": END,
        "rejected": "analyze_question"
    }
)
search_builder.add_edge("llm_fallback", END)

# --- 최종 그래프 컴파일 ---
memory = MemorySaver()
legal_rag_agent = search_builder.compile(checkpointer=memory, interrupt_before=["human_review"])