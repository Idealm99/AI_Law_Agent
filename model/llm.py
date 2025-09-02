# model/llm.py

from langchain_openai import ChatOpenAI
from schemas.schema import ToolSelectors

# 기본 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

# 라우팅을 위한 구조화된 출력 LLM
structured_llm_tool_selector = llm.with_structured_output(ToolSelectors)