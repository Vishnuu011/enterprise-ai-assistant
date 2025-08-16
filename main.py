from src.enterprise_ai_assistant.utils.utils import StructuredAndUnstructuredFileLoader
from src.enterprise_ai_assistant.vectorstore.rag_pipeline import VectorStoreRetrivers

#file_path: str = r"C:\Users\VISHNU\Desktop\resume_project_1\enterprise-ai-assistant\data" 
#document = StructuredAndUnstructuredFileLoader(file_path=file_path).file_loader_fun()
#print(document)

retriver = VectorStoreRetrivers(
    persist_directory="chroma-store",
    collection_name="company-docs"
).initialized_retriver_fun()

query = "what is alteration of memorandum"

r_doc = retriver.get_relevant_documents(query)
print(r_doc)
