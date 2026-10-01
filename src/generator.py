"""提示词 + LLM + OutputParser，组装成一条 Chain。

这个文件集中体现了你学过的：ChatPromptTemplate / Runnable（LCEL 管道符）/ StrOutputParser。
"""
from langchain_core.output_parsers import StrOutputParser   #输出解析工具
from langchain_core.prompts import ChatPromptTemplate       #构建提示词
from langchain_ollama import ChatOllama                     #调用ollama模型

from config import LLM_MODEL, OLLAMA_BASE_URL

SYSTEM_PROMPT = """你是一个企业知识库问答助手。只根据提供的参考内容回答问题。

规则：
1. 答案中的每个关键论断都要标注来源编号，如 [1]、[2]，编号对应参考内容前的方括号。
2. 如果参考内容不足以回答问题，明确回答"知识库中没有找到相关内容"，禁止编造。
3. 用简体中文回答，简洁、直接，不超过 300 字。"""


def build_chain():
    """返回一条 LCEL Chain: prompt | llm | parser

    调用方式: chain.invoke({"context": "...", "question": "..."}) -> str
    """
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,  # 知识库问答要稳，不要创造性
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "参考内容：\n{context}\n\n问题：{question}"),
        ]
    )
    return prompt | llm | StrOutputParser()
