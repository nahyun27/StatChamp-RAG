import os
from typing import List, Tuple
from langchain_community.chat_models import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, format_document
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from operator import itemgetter
from pydantic import BaseModel, Field

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class RAGModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data()
        self.vector_store = self.create_vector_store()
        self.llm = ChatOllama(model="llama3:latest")
        self.question_prompt = ChatPromptTemplate.from_template(
            "You are a baseball player analyst. Don't talk too hard but make it pleasant. {data}"
        )
        self.answer_prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}"
        )

    def load_data(self):
        data = pd.read_csv(self.data_path)
        data[['last_name', 'first_name']] = data['last_name, first_name'].str.split(', ', expand=True)
        data['text'] = data.apply(lambda row: f"Player: {row['first_name']} {row['last_name']}, Year: {row['year']}, PA: {row['pa']}, K%: {row['k_percent']}, BB%: {row['bb_percent']}, wOBA: {row['woba']}, xwOBA: {row['xwoba']}, LA Sweet-Spot %: {row['sweet_spot_percent']}, Barrel batted rate: {row['barrel_batted_rate']}, Hard Hit %: {row['hard_hit_percent']}, EV50: {row['avg_best_speed']}, Adjusted EV: {row['avg_hyper_speed']}, Whiff %: {row['whiff_percent']}, Swing %: {row['swing_percent']}", axis=1)
        return data

    def create_vector_store(self):
        loader = DataFrameLoader(self.data, page_content_column='text')
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(loader.load(), embeddings)
        return vector_store

    def predict(self, chat_history, question):
        data_str = self._format_chat_history(chat_history)
        
        # 시스템 메시지와 사용자 질문 메시지를 생성합니다.
        messages = [
            SystemMessage(content="You are a baseball player analyst. Don't talk too hard but make it pleasant."),
            HumanMessage(content=data_str),
            HumanMessage(content=question)
        ]
        
        # 질문에 대한 standalone 질문을 생성합니다.
        standalone_question_result = self.llm.invoke(messages)
        standalone_question = standalone_question_result.content
        
        # standalone 질문에 대한 답변을 생성합니다.
        standalone_answer_result = self.llm.invoke([HumanMessage(content=standalone_question)])
        standalone_answer = standalone_answer_result.content
        
        # 생성된 질문을 사용하여 벡터 스토어에서 유사한 문서를 검색합니다.
        search_results = self.vector_store.similarity_search(standalone_question, k=1)
        context = "\n\n".join([result.page_content for result in search_results])
        print('\n\n', search_results)

        # 컨텍스트와 질문을 사용하여 답변을 생성합니다.
        answer_messages = [
            SystemMessage(content="Answer the question based only on the following context:"),
            HumanMessage(content=context),
            HumanMessage(content=f"Question: {standalone_question}")
        ]
        
        answer_prompt_result = self.llm.invoke(answer_messages)
        answer = answer_prompt_result.content

        # What is the barrel percent of the Kyle Tucker in 2024?
        
        return {"standalone_answer": standalone_answer, "final_answer": answer}

    def _combine_documents(self, docs, document_prompt="{page_content}", document_separator="\n\n"):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    def _format_chat_history(self, chat_history):
        buffer = ""
        for dialogue_turn in chat_history:
            human = "Human: " + dialogue_turn[0]
            ai = "Assistant: " + dialogue_turn[1]
            buffer += "\n" + "\n".join([human, ai])
        return buffer

def initialize(data_path):
    return RAGModel(data_path)

def predict(model, chat_history, question):
    return model.predict(chat_history, question)
