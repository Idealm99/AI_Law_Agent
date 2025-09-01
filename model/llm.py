from model.tools.langchain_tools import tool_class 
from langchain_openai import ChatOpenAI
from pprint import pprint
from langchain_core.tools import tool

class openai :
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
        self.personal_law_search=tool_class.personal_law_search()
        self.labor_law_search =tool_class.labor_law_search()
        self.housing_law_search = tool_class.housing_law_search()
        self.web_search=tool_class.web_search()
        self.llm_with_tools = self.llm.bind_tools([self.personal_law_search, self.labor_law_search, self.housing_law_search, self.web_search])

    def invoke(self,query):
        ai_msg = self.llm_with_tools.invoke(query)
        pprint(ai_msg)
        print("-" * 100)

        pprint(ai_msg.content)
        print("-" * 100)

        pprint(ai_msg.tool_calls)
        print("-" * 100)


# 사용 예시
if __name__ == "__main__":
    
    # 개인정보보호법 컬렉션 검색
    openai.invoke(
        query="개인정보가 유출되면 어떻게 해야 하나요?"
    )
    
    # 근로기준법 컬렉션 검색
    openai.invoke(
        query="미성년자 근로계약"
    )

    # 주택임대차보호법 컬렉션 검색
    openai.invoke(
        query="계약 갱신 요구권"
    )