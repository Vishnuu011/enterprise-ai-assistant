import os, sys
from src.enterprise_ai_assistant.vectorstore.rag_pipeline import VectorStoreRetrivers

from langchain_core.messages import (
    HumanMessage,
    BaseMessage
)
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_experimental.agents import create_pandas_dataframe_agent

from src.enterprise_ai_assistant.data_models.models import (
    VectorStore,
    SearchEngine,
    SQLDatabaseAgent,
    DataframeCSVAgent,
    RelevanceGrader,
    AnswerGrader,
    HallucinationGrader
)
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence
from src.enterprise_ai_assistant.constant import *

from src.enterprise_ai_assistant.prompts.prompt_template import (
    router_prompt_template,
    relevance_system_prompt_template,
    rag_template,
    hallucination_system_prompt_template,answer_system_prompt_template,
    fallback_prompts
)
from typing import Optional, List, Literal
from dotenv import load_dotenv
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

load_dotenv()



def router_function(model_name: str) -> Optional[RunnableSequence]:

    try:
        llm = ChatGroq(
            model=model_name,
            temperature=TEMPERATURE
        )

        router_prompt = ChatPromptTemplate.from_template(
            router_prompt_template
        )

        question_router = (
            router_prompt
            | llm.bind_tools(
                tools=[
                    VectorStore,
                    SearchEngine,
                    DataframeCSVAgent,
                    SQLDatabaseAgent
                ]
            )
        )

        return question_router   
        
    except Exception as e:
        print(f"error occured in: {e}")

        logger.error(e)


def grader_function(model_name: str) -> Optional[RunnableSequence]:

    try:
        llm = ChatGroq(
            model=model_name,
            temperature=TEMPERATURE
        )

        relevance_prompt = ChatPromptTemplate.from_template(
            relevance_system_prompt_template + "\n\ncontext: {context}\n\nquery: {query}"
        )

        relevance_chain = (
            relevance_prompt
            | llm.with_structured_output(
                RelevanceGrader,
                method="json_mode"
            )
        )

        return relevance_chain
    except Exception as e:
        print(f"error occured in: {e}")
  
        logger.error(e)


def hallucination_function(model_name: str) -> Optional[RunnableSequence]:

    try:
        llm = ChatGroq(
            model=model_name,
            temperature=TEMPERATURE
        )

        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", hallucination_system_prompt_template),
                ("human", "context: {context}\n\nllm_response: {response}")
            ]
        )

        hallucination_chain = (
            hallucination_prompt
            | llm.with_structured_output(
                HallucinationGrader,
                method="json_mode"
            )
        )

        return hallucination_chain
        
    except Exception as e:
        print(f"error occured in: {e}")
        logger.error(e)



def answer_grader_function(model_name: str) -> Optional[RunnableSequence]:

    try:
        llm = ChatGroq(
            model=model_name,
            temperature=TEMPERATURE
        )

        answer_grader_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", answer_system_prompt_template),
                ("human", "query: {query}\n\nanswer: {response}")
            ]
        )

        answer_chain = (
            answer_grader_prompt
            | llm.with_structured_output(
                AnswerGrader,
                method="json_mode"
            )
        )

        return answer_chain
        
    except Exception as e:
        print(f"error occured in: {e}")
        logger.error(e)



def rag_function(model_name: str) -> Optional[RunnableSequence]:

    try:
        llm = ChatGroq(
            model=model_name,
            temperature=TEMPERATURE
        )

        rag_prompt = ChatPromptTemplate.from_template(
            rag_template
        )

        rag_chain = (
            rag_prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain
    
    except Exception as e:
        print(f"error occured in: {e}")
        logger.error(e)



def format_history(msgs: List[BaseMessage]) -> str:
    return "\n".join([f'human: {m.content}' if isinstance(m, HumanMessage) else f'AI: {m.content}' for m in msgs])




def fallback_function(model_name: str) -> Optional[RunnableSequence]:

    try:
        llm = ChatGroq(
            model=model_name,
            temperature=TEMPERATURE
        )

        fallback_prompt = ChatPromptTemplate.from_messages(
            fallback_prompts
        )

        fallback_chain = (
            {
                "chat_history": lambda x: format_history(x.get("chat_history", [])),
                "query": itemgetter("query"),
            }
            | fallback_prompt
            | llm
            | StrOutputParser()    
        )

        return fallback_chain
    
    except Exception as e:
        print(f"error occured in: {e}")
        logger.error(e)



def sql_agent_function(model: str, sql_data_folder: str) -> Optional[AgentExecutor]:

    try:
        #sql_data_folder = "sql"
        for file_name in os.listdir(sql_data_folder):

            file_path = os.path.join(sql_data_folder, file_name)
        
        db = SQLDatabase.from_uri(f"sqlite:///{file_path}")

        llm = ChatGroq(
            model=model,
            temperature=TEMPERATURE
        )

        sql_agent = create_sql_agent(
            llm=llm,
            db=db,
            agent_type="openai-tools",
            verbose=False
        )

        return sql_agent
    
    except Exception as e:
        print(f"error occured in: {e}")
        logger.error(e)     



def pandas_dataframe_function(model: str, csv_folder: str) -> Optional[AgentExecutor]:

    try:
        for file_name in os.listdir(csv_folder):

            file_path = os.path.join(csv_folder, file_name)

        df = pd.read_csv(file_path)    

        llm = ChatGroq(
            model=model,
            temperature=TEMPERATURE
        )  

        pandas_agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            agent_type="openai-tools",
            verbose=False,
            allow_dangerous_code=True

        )  

        return pandas_agent
    
    except Exception as e:
        print(f"error occured in: {e}")
        logger.error(e)     




def web_search_engine() -> TavilySearchResults:

    try:
        search_engine = TavilySearchResults()
        return search_engine
    except Exception as e:
        print(f"errror occured in: {e}")
        logger.error(e)
        return None

        