#利用stramlit构建app

# 1.导入必要的库
import streamlit as st
import os 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch,RunnablePassthrough
from langchain_community.vectorstores import Chroma
from zhipuai_embedding import ZhipuAIEmbedding
from zhipuai_llm import ZhipuaiLLM

# 2.定义get_retriever函数 ，该函数返回一个检索器
def get_retriever():
    embedding = ZhipuAIEmbedding()
    persist_dir = '../data_base/vector_db/chroma'
    vectordb = Chroma(
        embedding_function = embedding,
        persist_directory = persist_dir
    )
    return vectordb.as_retriever()

# 3.定义combine_docs函数，处理检索器返回的文本
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])

# 4.定义get_qa_history_chain,该函数返回一个检索问答链
def get_qa_history_chain():
    retriever = get_retriever()
    llm = ZhipuaiLLM(
        model_name="glm-4",
        temperature=0.1
    )
    condense_question_system_template=(
        "请根据聊天记录完善用户的问题。"
        "如果用户的问题不需要完善则直接返回该问题。"
    )
    condense_question_prompt = ChatPromptTemplate.from_messages([
        ("system", condense_question_system_template),
        ("placeholder","{chat_history}"),
        ("human","{input}"),
    ])
    #检索链
    retriever_docs = RunnableBranch(
        (lambda x: not x.get("chat_history",False),(lambda x:x["input"])|retriever,),
        condense_question_prompt|llm|StrOutputParser()|retriever,
        )
    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt =ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        ("placeholder","{chat_history}"),
        ("human","{input}"),
    ])
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        |qa_prompt
        |llm
        |StrOutputParser()
    )
    qa_history_chain = RunnablePassthrough().assign(context=retriever_docs).assign(answer=qa_chain)
    return qa_history_chain

  
# 5. 定义gen_response函数，接受检索问答链、用户输入和聊天历史
def gen_response(chain,input,chat_history):
    response = chain.strea(
        {
            "input",input,
            "chat_history",chat_history
        }
    )
    for res  in response:
        if "answer" in res.key():
            yield res["answer"]

# 6.定义main函数，指定显示效果与逻辑
def main():
    st.markdown('### 🦜🔗 动手学大模型应用开发')
    #存储对话历史
    if "messages" not in st.session_state:
        st.session_stata.messages = []
    #存储问答检索链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
    #聊天容器
    messages = st.container(height=550)
    #遍历聊天记录并打印对话（带头像）
    for message in st.session_state.messages:
        with messages.chat_message(message[0]):
            st.write(message[1])#打印内容
    if prompt := st.chat_input("Say something"):
        #记录用户输入
        st.session_state.messages.append(("human",prompt))
        #显示用户输入
        with messages.chat_message("human"):
            st.write(prompt) 
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
            st.session_state.messages.append(("ai",output))
            