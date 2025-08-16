from typing import (
    TypedDict, 
    List, 
    Literal, 
    Dict, 
    Any
)

from src.enterprise_ai_assistant.components.com_fun import (
    router_function,
    grader_function,
    hallucination_function,
    answer_grader_function,
    rag_function,
    fallback_function,
    sql_agent_function,
    pandas_dataframe_agent_function,
    web_search_engine
)

from src.enterprise_ai_assistant.constant import *

from src.enterprise_ai_assistant.data_models.models import AgentState

from src.enterprise_ai_assistant.vectorstore.rag_pipeline import VectorStoreRetrivers
from langchain.schema import Document

from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
import warnings
import logging
warnings.filterwarnings("ignore")

logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)




class CyclicGraphsWorkflow:

    def __init__(self):

        pass

    def retrieve_node(self, state: AgentState) -> Dict[str, Any]:

        try:
            retriever = VectorStoreRetrivers(
                persist_directory="chroma-store",
                collection_name="company-docs"
            ).initialized_retriver_fun()

            query = state["query"]
            documents = retriever.invoke(query) if retriever else []
            return {"documents": documents}
        except Exception as e:
            print(f"[Retrieve Node Error]: {e}")
            return {"documents": []}


    def web_search_node(self, state: AgentState) -> Dict[str, Any]:

        try:
            tavily_search = web_search_engine()
            query = state["query"]
            results = tavily_search.invoke(query) if tavily_search else []

            documents = [
                Document(page_content=r.get("content", ""), metadata={"source": r.get("url", "")})
                for r in results
            ]
            return {"documents": documents}
        except Exception as e:
            print(f"[Web Search Error]: {e}")
            return {"documents": []}


    def filter_documents_node(self, state: AgentState) -> Dict[str, Any]:

        try:
            query = state["query"]
            documents = state.get("documents", [])

            relevance_chain = grader_function(model_name=MODEL_1)
            filtered = []

            for i, doc in enumerate(documents, start=1):
                grade = relevance_chain.invoke({"query": query, "context": doc.page_content})
                if grade.grade == "relevant":
                    print(f"--- CHUNK {i}: RELEVANT ---")
                    filtered.append(doc)
                else:
                    print(f"--- CHUNK {i}: NOT RELEVANT ---")

            return {"documents": filtered}
        except Exception as e:
            print(f"[Filter Docs Error]: {e}")
            return {"documents": []}


    def rag_node(self, state: AgentState) -> Dict[str, Any]:

        try:
            query = state["query"]
            docs = state.get("documents", [])
            context = "\n\n".join([d.page_content for d in docs]) if docs else ""

            rag_chain = rag_function(model_name=MODEL_1)
            generation = rag_chain.invoke({"query": query, "context": context})

            return {"generation": generation}
        except Exception as e:
            print(f"[RAG Node Error]: {e}")
            return {"generation": "RAG failed."}


    def sql_agent_node(self, state: AgentState) -> Dict[str, Any]:

        try:
            query = state["query"]
            sql_agent = sql_agent_function(
                model=MODEL_2,
                sql_data_folder=r"C:\Users\VISHNU\Desktop\resume_project_1\enterprise-ai-assistant\sql"
            )

            resp = sql_agent.invoke({"input": query}) if sql_agent else {}
            generation = resp.get("output") if isinstance(resp, dict) else str(resp)
            return {"generation": generation}
        except Exception as e:
            print(f"[SQL Agent Error]: {e}")
            return {"generation": f"Error running SQL agent: {e}"}

 
    def pandas_agent_node(self, state: AgentState) -> Dict[str, Any]:

        try:
            query = state["query"]
            agent_panda_sql = pandas_dataframe_agent_function(
                model=MODEL_1,
                csv_folder=r"C:\Users\VISHNU\Desktop\resume_project_1\enterprise-ai-assistant\csv_data"
            )

            resp = agent_panda_sql.invoke({"input": query}) if agent_panda_sql else {}
            generation = resp.get("output") if isinstance(resp, dict) else str(resp)

            return {"generation": generation}
        
        except Exception as e:
            print(f"[Pandas Agent Error]: {e}")
            return {"generation": f"Error running Pandas agent: {e}"}


    def fallback_node(self, state: AgentState) -> Dict[str, Any]:

        try:
            fallback_chain = fallback_function(model_name=MODEL_1)

            generation = fallback_chain.invoke({
                "query": state["query"],
                "chat_history": state.get("chat_history", [])
            })

            return {"generation": generation}
        
        except Exception as e:
            print(f"[Fallback Error]: {e}")
            return {"generation": "Fallback failed."}


    def question_router_node(self, state: AgentState) -> str:
        try:
            query = state["query"]
            question_router = router_function(model_name=MODEL_1)

            response = question_router.invoke({"query": query})

            tool_calls = getattr(response, "additional_kwargs", {}).get("tool_calls", [])

            if not tool_calls:
                print("--- Router made no tool call ---")
                return "llm_fallback"

            route = tool_calls[0]["function"]["name"]
            print(f"--- Routing to: {route} ---")

            return route if route in {
                "VectorStore",
                "SearchEngine",
                "SQLDatabaseAgent",
                "DataframeCSVAgent"
            } else "llm_fallback"
        
        except Exception as e:
            print(f"[Router Error]: {e}")

            return "llm_fallback"


    def should_generate(self, state: AgentState) -> str:
        try:
            docs = state.get("documents", [])

            if not docs:
                print("--- No relevant documents; try SearchEngine ---")
                return "SearchEngine"
            print("--- Some documents relevant; proceed to RAG ---")

            return "generate"
        
        except Exception as e:
            print(f"[Should Generate Error]: {e}")
            return "SearchEngine"

  
    def quality_gate_node(self, state: AgentState) -> Dict[str, str]:
        
        try:
            llm_response = state.get("generation", "")

            docs = state.get("documents", [])

            ctx = "\n\n".join([d.page_content for d in docs]) if docs else ""

            answer_chain = answer_grader_function(model_name=MODEL_1)
            hallucination_chain = hallucination_function(model_name=MODEL_1)

            if not ctx.strip():
                print("--- No context; skipping hallucination check ---")
                ans = answer_chain.invoke({"response": llm_response, "query": state["query"]})
                return {"quality_gate_decision": "useful" if ans.grade == "yes" else "not useful"}

            halluc = hallucination_chain.invoke({"response": llm_response, "context": ctx})
            if halluc.grade == "no":
                ans = answer_chain.invoke({"response": llm_response, "query": state["query"]})
                return {"quality_gate_decision": "useful" if ans.grade == "yes" else "not useful"}

            print("--- Hallucination detected ---")

            return {"quality_gate_decision": "regenerate"}
        
        except Exception as e:
            print(f"[Quality Gate Error]: {e}")
            return {"quality_gate_decision": "not useful"}


    def build_graph_workflow(self) -> CompiledStateGraph[AgentState]:

        try:
            workflow = StateGraph(AgentState)

            workflow.add_node("VectorStore", self.retrieve_node)
            workflow.add_node("SearchEngine", self.web_search_node)
            workflow.add_node("filter_docs", self.filter_documents_node)
            workflow.add_node("rag", self.rag_node)
            workflow.add_node("SQLDatabaseAgent", self.sql_agent_node)
            workflow.add_node("DataframeCSVAgent", self.pandas_agent_node)
            workflow.add_node("fallback", self.fallback_node)
            workflow.add_node("quality_gate", self.quality_gate_node)

            workflow.set_conditional_entry_point(
                self.question_router_node,
                {
                    "llm_fallback": "fallback",
                    "VectorStore": "VectorStore",
                    "SearchEngine": "SearchEngine",
                    "SQLDatabaseAgent": "SQLDatabaseAgent",
                    "DataframeCSVAgent": "DataframeCSVAgent",
                },
            )

            workflow.add_edge("VectorStore", "filter_docs")
            workflow.add_edge("SearchEngine", "filter_docs")
            workflow.add_conditional_edges(
                "filter_docs",
                self.should_generate,
                {"SearchEngine": "SearchEngine", "generate": "rag"},
            )

            workflow.add_edge("SQLDatabaseAgent", "quality_gate")
            workflow.add_edge("DataframeCSVAgent", "quality_gate")
            workflow.add_edge("rag", "quality_gate")

            workflow.add_conditional_edges(
                "quality_gate",
                lambda state: state.get("quality_gate_decision"),
                {"useful": END, "not useful": "SearchEngine", "regenerate": "rag"},
            )

            workflow.add_edge("fallback", END)

            return workflow.compile()
        except Exception as e:
            print(f"[Build Graph Error]: {e}")
            return None

