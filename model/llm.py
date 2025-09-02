from model.tools.langchain_tools import tool_class 
from langchain_openai import ChatOpenAI
from pprint import pprint
from langchain_core.tools import tool
from schemas.schema import RagState
import os
from dotenv import load_dotenv
load_dotenv()
# from schemas.schema import pt
class openai:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
        # 툴 인스턴스가 아니라 툴 함수/클래스 자체를 전달
        self.llm_with_tools = self.llm.bind_tools([
            tool_class.personal_law_search,
            tool_class.labor_law_search,
            tool_class.housing_law_search,
            tool_class.web_search
        ])

    def invoke(self,query):
        
    
    
    
        # llm=self.llm_with_tools.with_structured_output(RagState)
        ai_msg=self.llm_with_tools.invoke(query)
            
  
        
        # ai_msg=self.llm_with_tools.invoke(query)
        pprint(ai_msg)
        print("-" * 100)

        pprint(ai_msg.content)
        print("-" * 100)

        pprint(ai_msg.tool_calls)
        print("-" * 100)

        # pprint(ai_msg.get('content'))
        # print("-" * 100)

        # pprint(ai_msg.get('tool_calls'))
        # print("-" * 100)

# 사용 예시
if __name__ == "__main__":
    agent = openai()
    # 개인정보보호법 컬렉션 검색
    agent.invoke(
        query="개인정보가 유출되면 어떻게 해야 하나요?"
    )
    
    # 근로기준법 컬렉션 검색
    agent.invoke(
        query="미성년자 근로계약"
    )

    # 주택임대차보호법 컬렉션 검색
    agent.invoke(
        query="계약 갱신 요구권"
    )