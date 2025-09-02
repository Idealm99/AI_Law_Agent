# model/graph/nodes.py

from typing import Literal, List
import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from schemas.schema import (
    ResearchAgentState, PersonalRagState, LaborRagState, HousingRagState, SearchRagState,
    ExtractedInformation, RefinedQuestion
)
from model.prompts.prompt import (
    get_extract_prompt, get_rewrite_prompt, get_answer_prompt,
    route_prompt, rag_prompt, fallback_prompt
)
from model.tools.langchain_tools import (
    personal_law_search, labor_law_search, housing_law_search, web_search, tools
)
from model.llm import llm, structured_llm_tool_selector
from langgraph.prebuilt import create_react_agent

# --- Corrective RAG 노드 생성 함수 ---
def create_rag_nodes(law_name: str, search_tool: callable, state_type: type):
    
    def retrieve_documents(state: state_type) -> state_type:
        print(f"---{law_name} 문서 검색---")
        query = state.get("rewritten_query", state["question"])
        docs = search_tool.invoke(query)
        return {"documents": docs}

    def extract_and_evaluate_information(state: state_type) -> state_type:
        print(f"---{law_name} 정보 추출 및 평가---")
        extracted_strips = []
        extract_prompt = get_extract_prompt(law_name)
        extract_llm = llm.with_structured_output(ExtractedInformation)
        
        for doc in state["documents"]:
            extracted_data = extract_llm.invoke({
                "question": state["question"],
                "document_content": doc.page_content
            })
            if extracted_data.query_relevance < 0.8:
                continue
            for strip in extracted_data.strips:
                if strip.relevance_score > 0.7 and strip.faithfulness_score > 0.7:
                    extracted_strips.append(strip)
        
        return {
            "extracted_info": extracted_strips,
            "num_generations": state.get("num_generations", 0) + 1
        }

    def rewrite_query(state: state_type) -> state_type:
        print(f"---{law_name} 쿼리 재작성---")
        rewrite_prompt = get_rewrite_prompt(law_name)
        extracted_info_str = "\\n".join([strip.content for strip in state["extracted_info"]])
        rewrite_llm = llm.with_structured_output(RefinedQuestion)
        
        response = rewrite_llm.invoke({
            "question": state["question"],
            "extracted_info": extracted_info_str
        })
        
        return {"rewritten_query": response.question_refined}

    def generate_node_answer(state: state_type) -> state_type:
        print(f"---{law_name} 답변 생성---")
        source_example = f"{law_name} 제15조" if "법" in law_name else "블로그 (www.example.com)"
        answer_prompt = get_answer_prompt(law_name, source_example)
        extracted_info_str = "\\n".join([f"내용: {s.content}\\n출처: {s.source}" for s in state["extracted_info"]])
        
        node_answer = answer_prompt.invoke({
            "question": state["question"],
            "extracted_info": extracted_info_str
        })
        
        return {"node_answer": node_answer.content}

    def should_continue(state: state_type) -> Literal["continue", "end"]:
        if state["num_generations"] >= 2:
            return "end"
        if state.get("extracted_info") and len(state["extracted_info"]) >= 1:
            return "end"
        return "continue"

    return retrieve_documents, extract_and_evaluate_information, rewrite_query, generate_node_answer, should_continue

# --- Supervisor 노드 ---
def analyze_question_tool_search(state: ResearchAgentState):
    print("---질문 분석 및 라우팅---")
    question = state["question"]
    result = structured_llm_tool_selector.invoke(route_prompt.format(question=question))
    datasources = [tool.tool for tool in result.tools]
    return {"datasources": datasources}

def route_datasources_tool_search(state: ResearchAgentState) -> List[str]:
    return list(set(state['datasources']))

def answer_final(state: ResearchAgentState) -> ResearchAgentState:
    print("---최종 답변 생성---")
    question = state["question"]
    documents = state.get("answers", [])
    documents_text = "\\n\\n".join(documents)
    
    rag_chain = rag_prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"documents": documents_text, "question": question})
    return {"final_answer": generation, "question": question}

def llm_fallback(state: ResearchAgentState) -> ResearchAgentState:
    print("---Fallback 답변---")
    question = state["question"]
    llm_chain = fallback_prompt | llm | StrOutputParser()
    generation = llm_chain.invoke({"question": question})
    return {"final_answer": generation, "question": question}

# --- 평가 및 HITL 노드 ---
answer_reviewer = create_react_agent(llm, tools=tools, state_modifier=evaluation_prompt)

def evaluate_answer_node(state: ResearchAgentState):
    print("---답변 평가---")
    question = state["question"]
    final_answer = state["final_answer"]

    messages = [HumanMessage(content=f'\"\"\"[질문]\\n\\{question}\\n\\n[답변]\\n{final_answer}\"\"\"')]
    response = answer_reviewer.invoke({"messages": messages})
    response_dict = json.loads(response['messages'][-1].content)

    return {"evaluation_report": response_dict, "question": question, "final_answer": final_answer}

def human_review_node(state: ResearchAgentState):
    # 이 노드는 LangGraph가 중단되는 지점입니다. 실제 사용자 입력은 Gradio 인터페이스에서 처리됩니다.
    pass