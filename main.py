from src.enterprise_ai_assistant.utils.utils import StructuredAndUnstructuredFileLoader
from src.enterprise_ai_assistant.vectorstore.rag_pipeline import VectorStoreRetrivers
from src.enterprise_ai_assistant.stategraph.nodes_and_workflow import CyclicGraphsWorkflow
from IPython.display import display, Image
from src.enterprise_ai_assistant.data_models.models import AgentState

#file_path: str = r"C:\Users\VISHNU\Desktop\resume_project_1\enterprise-ai-assistant\data" 
#document = StructuredAndUnstructuredFileLoader(file_path=file_path).file_loader_fun()
#print(document)

app = CyclicGraphsWorkflow().build_graph_workflow()
# #query = "given document What is cross-validation, and why is it important?"

# while True:
#     user_query = input("\nEnter your question (or 'exit' to quit): ")
#     if user_query.lower() == "exit":
#         break

#     init_state: AgentState = {
#         "query": user_query,
#         "chat_history": [],
#         "generation": "",
#         "documents": [],
#     }

#     # Stream the graph (shows intermediate steps)
#     for step in app.stream(init_state):
#         pass   # you can add logging here if you want to inspect steps

#     # Final state after execution
#     final_state = app.invoke(init_state)

#     print("\n=== FINAL ANSWER ===")
#     print(final_state.get("generation", ""))

with open("graph.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())


