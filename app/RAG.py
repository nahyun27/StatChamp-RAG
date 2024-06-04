# app/RAG.py
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
    def __init__(self, baseball_data_path, soccer_players_data_path, soccer_stats_data_path, soccer_stats_data_entire_path, sport):
        self.baseball_data_path = baseball_data_path
        self.soccer_players_data_path = soccer_players_data_path
        self.soccer_stats_data_path = soccer_stats_data_path
        self.soccer_stats_data_entire_path = soccer_stats_data_entire_path
        self.sport = sport
        self.data = self.load_data()
        print(f"Loaded data for {self.sport}:")
        print(self.data.head())  # 데이터의 처음 몇 줄을 출력
        self.vector_store = self.create_vector_store()
        self.llm = ChatOllama(model="llama3:latest")
        self.question_prompt = ChatPromptTemplate.from_template(
            f"You are a {self.sport} player analyst. Don't talk too hard but make it pleasant. {{data}}"
        )
        self.answer_prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}"
        )

    def load_data(self):
        if self.sport == 'baseball':
            data = pd.read_csv(self.baseball_data_path)
            data[['last_name', 'first_name']] = data['last_name, first_name'].str.split(', ', expand=True)
            data['text'] = data.apply(lambda row: f"Player: {row['first_name']} {row['last_name']}, Year: {row['year']}, PA: {row['pa']}, K%: {row['k_percent']}, BB%: {row['bb_percent']}, wOBA: {row['woba']}, xwOBA: {row['xwoba']}, LA Sweet-Spot %: {row['sweet_spot_percent']}, Barrel%: {row['barrel_batted_rate']}, Hard Hit %: {row['hard_hit_percent']}, EV50: {row['avg_best_speed']}, Adjusted EV: {row['avg_hyper_speed']}, Whiff %: {row['whiff_percent']}, Swing %: {row['swing_percent']}", axis=1)
        elif self.sport == 'soccer':
            players_data = pd.read_csv(self.soccer_players_data_path)
            stats_data = pd.read_csv(self.soccer_stats_data_path)
            stats_data_entire = pd.read_csv(self.soccer_stats_data_entire_path)

            # 데이터 컬럼 이름 소문자 처리 및 공백 제거
            players_data.columns = players_data.columns.str.strip().str.replace(' ', '').str.lower()
            stats_data.columns = stats_data.columns.str.strip().str.replace(' ', '').str.lower()
            stats_data_entire.columns = stats_data_entire.columns.str.strip().str.replace(' ', '').str.lower()

            players_data['text'] = players_data.apply(lambda row: f"Player: {row['playername']}, Team: {row['teamname']}, Position: {row['position']}, Market Value: {row['marketvalue']}", axis=1)
            stats_data['text'] = stats_data.apply(lambda row: f"Player: {row['playername']}, Appearances: {row['appearances']}, Goals: {row['goals']}, Assists: {row['assists']}, Passes: {row['passes']}, Tackles: {row['tackles']}, Interceptions: {row['interceptions']}", axis=1)
            stats_data_entire['text'] = stats_data_entire.apply(lambda row: f"Player: {row['playername']}, Career Saves: {row['saves']}, Career Goals: {row['goals']}, Career Assists: {row['assists']}", axis=1)

            data = pd.concat([players_data[['text']], stats_data[['text']], stats_data_entire[['text']]], ignore_index=True)
        return data

    def create_vector_store(self):
        if self.data.empty:
            raise ValueError("No data available to create vector store.")  # 데이터가 비어 있을 때 예외 발생
        loader = DataFrameLoader(self.data, page_content_column='text')
        documents = loader.load()
        print("Loaded documents for vector store:")  # 디버깅 출력
        for doc in documents[:5]:  # 처음 몇 개의 문서를 출력하여 확인
            print(doc.page_content)  # 디버깅 출력
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store

    def predict(self, chat_history, question):
        data_str = self._format_chat_history(chat_history)
        
        # 시스템 메시지와 사용자 질문 메시지를 생성합니다.
        messages = [
            SystemMessage(content=f"You are a {self.sport} player analyst. Don't talk too hard but make it pleasant."),
            HumanMessage(content=data_str),
            HumanMessage(content=question)
        ]
        
        # standalone 질문에 대한 답변을 생성합니다.
        standalone_answer_result = self.llm.invoke(messages)
        standalone_answer = standalone_answer_result.content
        
        # 질문에 대한 컨텍스트 검색
        search_results = self.vector_store.similarity_search(question, k=10)
        context = "\n\n".join([result.page_content for result in search_results])
        
        # 컨텍스트와 질문을 사용하여 RAG 답변을 생성합니다.
        answer_messages = [
            SystemMessage(content="Answer the question based only on the following context:"),
            HumanMessage(content=context),
            HumanMessage(content=f"Question: {question}")
        ]
        
        answer_prompt_result = self.llm.invoke(answer_messages)
        answer = answer_prompt_result.content
        
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

def initialize(baseball_data_path, soccer_players_data_path, soccer_stats_data_path, soccer_stats_data_entire_path, sport):
    return RAGModel(baseball_data_path, soccer_players_data_path, soccer_stats_data_path, soccer_stats_data_entire_path, sport)

def predict(model, chat_history, question):
    return model.predict(chat_history, question)
