from langchain_community.chat_models import ChatOllama
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

class RAGModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data()
        self.vector_store = self.create_vector_store()
        self.llm = ChatOllama(model="llama3:latest")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a baseball player analyst. Don't talk too hard but make it pleasant."),
            ("assistant", "{data}"),
            ("user", "{input}")
        ])

    def load_data(self):
        # Load data from CSV, make sure to handle the encoding if necessary.
        data = pd.read_csv(self.data_path, encoding='cp949')
        # Apply transformation to format the data
        data['text'] = data.apply(
            lambda row: (
                f"Season: {row['Season']}, Date: {row['DateTime']}, Home Team: {row['HomeTeam']}, "
                f"Away Team: {row['AwayTeam']}, Full Time Home Goals: {row['FTHG']}, Full Time Away Goals: {row['FTAG']}, "
                f"Full Time Result: {row['FTR']}, Half Time Home Goals: {row['HTHG']}, Half Time Away Goals: {row['HTAG']}, "
                f"Half Time Result: {row['HTR']}, Referee: {row['Referee']}, Home Shots: {row['HS']}, Away Shots: {row['AS']}, "
                f"Home Shots on Target: {row['HST']}, Away Shots on Target: {row['AST']}, Home Corners: {row['HC']}, "
                f"Away Corners: {row['AC']}, Home Fouls: {row['HF']}, Away Fouls: {row['AF']}, "
                f"Home Yellow Cards: {row['HY']}, Away Yellow Cards: {row['AY']}, Home Red Cards: {row['HR']}, "
                f"Away Red Cards: {row['AR']}"
            ), axis=1
        )
        return data

    def create_vector_store(self):
        loader = DataFrameLoader(self.data, page_content_column='text')
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(loader.load(), embeddings)
        return vector_store

    def predict(self, query):
        search_results = self.vector_store.similarity_search(query, k=1)
        data_for_llm = "\n".join([result.page_content for result in search_results])
        response = self.prompt | self.llm
        output = response.invoke({"data": data_for_llm, "input": query}).content
        return output

def initialize(data_path):
    return RAGModel(data_path)

def predict(model, input_text):
    return model.predict(input_text)
