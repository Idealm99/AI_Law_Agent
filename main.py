# main.py

import gradio as gr
import uuid
from typing import List, Tuple
from dotenv import load_dotenv

from model.graph.langgraph import legal_rag_agent

# 환경 변수 로드
load_dotenv()

# 예시 질문
example_questions = [
    "사업장에서 CCTV를 설치할 때 주의해야 할 법적 사항은 무엇인가요?",
    "전월세 계약 갱신 요구권의 행사 기간과 조건은 어떻게 되나요?",
    "개인정보 유출 시 기업이 취해야 할 법적 조치는 무엇인가요?",
]

# 챗봇 클래스
class ChatBot:
    def __init__(self):
        self.thread_id = str(uuid.uuid4())
        self.user_decision_pending = False

    def _get_config(self):
        return {"configurable": {"thread_id": self.thread_id}}

    def _process_stream_and_get_response(self, stream, initial_message):
        final_answer = initial_message
        for chunk in stream:
            if "generate_answer" in chunk:
                final_answer = chunk["generate_answer"].get("final_answer", final_answer)
        return final_answer
    
    def chat(self, message: str, history: List[Tuple[str, str]]) -> str:
        print(f"Thread ID: {self.thread_id}")
        config = self._get_config()
        
        try:
            if not self.user_decision_pending:
                # Breakpoint까지 실행
                inputs = {"question": message}
                legal_rag_agent.invoke(inputs, config=config)
                
                # Breakpoint에서 현재 상태 가져오기
                current_state = legal_rag_agent.get_state(config)
                final_answer = current_state.values.get("final_answer", "답변을 생성 중입니다...")
                eval_report = current_state.values.get('evaluation_report', {})
                
                response = f"""**생성된 답변:**\n{final_answer}\n\n---\n**자체 평가:**\n- **점수:** {eval_report.get('total_score', 'N/A')}/60\n- **평가:** {eval_report.get('brief_evaluation', 'N/A')}\n\n**이 답변이 마음에 드시나요? (y/n)**"""
                
                self.user_decision_pending = True
                return response
            else:
                user_input = message.lower().strip()
                if user_input == 'y':
                    decision = "approved"
                    legal_rag_agent.update_state(config, {"user_decision": decision})
                    final_stream = legal_rag_agent.stream(None, config=config)
                    final_response = self._process_stream_and_get_response(final_stream, "승인되었습니다.")
                    self.user_decision_pending = False
                    # 새로운 대화를 위해 스레드 ID 변경
                    self.thread_id = str(uuid.uuid4()) 
                    return "답변이 승인되었습니다. 새로운 질문을 해주세요."
                elif user_input == 'n':
                    decision = "rejected"
                    legal_rag_agent.update_state(config, {"user_decision": decision})
                    
                    # 거부 후 다시 실행
                    legal_rag_agent.invoke(None, config=config)
                    
                    current_state = legal_rag_agent.get_state(config)
                    final_answer = current_state.values.get("final_answer", "답변을 재작성 중입니다...")
                    eval_report = current_state.values.get('evaluation_report', {})
                    
                    response = f"""**재작성된 답변:**\n{final_answer}\n\n---\n**자체 평가:**\n- **점수:** {eval_report.get('total_score', 'N/A')}/60\n- **평가:** {eval_report.get('brief_evaluation', 'N/A')}\n\n**이 답변이 마음에 드시나요? (y/n)**"""
                    
                    self.user_decision_pending = True
                    return response
                else:
                    return "올바른 입력을 해주세요. (y 또는 n)"

        except Exception as e:
            print(f"Error occurred: {e}")
            self.user_decision_pending = False
            return "죄송합니다. 오류가 발생했습니다. 다시 시도해 주세요."

# 챗봇 인스턴스 생성 및 Gradio 인터페이스 실행
chatbot_instance = ChatBot()
demo = gr.ChatInterface(
    fn=chatbot_instance.chat,
    title="⚖️ LangGraph 기반 법률 AI 에이전트",
    description="주택임대차보호법, 근로기준법, 개인정보보호법에 대해 질문해보세요.",
    examples=example_questions,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()