import os, sys
import warnings

from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.enterprise_ai_assistant.utils.utils import StructuredAndUnstructuredFileLoader

from typing import List, Optional, Literal
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever



import logging

logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

warnings.filterwarnings("ignore")



class VectorStoreRetrivers:

    def __init__(self, persist_directory: str, collection_name: str):

        self.persist_directory = persist_directory
        self.collection_name = collection_name

    def chunking_fun(self) -> List[Document]:

        try:
            logger.info("chunking fun is started ...")
            logger.info("data loading started...")
            documents = StructuredAndUnstructuredFileLoader(
                file_path=os.path.join("data")
            ).file_loader_fun()
            logger.info("data has been loaded....")
            logger.info("chunking is started...")
            spliter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 200
            )
            logger.info("spliting document is running...")

            chunks = spliter.split_documents(documents=documents)
            logger.info("document splited .....")
            logger.info("chunking fun exited....")
            return chunks
        except Exception as e:
            print(f"error occured in: {e}")
            logger.error(f"error occured in: {e}")

    def embedding_function(self) -> HuggingFaceEmbeddings:

        try:
            logger.info("initialized Hugging face embedings....")
            embedd_fun = HuggingFaceEmbeddings()
            return embedd_fun
        except Exception as e:
            print(f"error occured in: {e}")
            logger.info(f"error occured in: {e}")     

    def vector_store_fun(self, documents: list[Document], 
                         embedd_fun: HuggingFaceEmbeddings) -> Optional[Chroma]:

        try:
            logger.info("initialized chroma db vector store....")
            vector_db = Chroma.from_documents(
                documents=documents,
                embedding=embedd_fun,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            logger.info("chroma vector store created....")
            logger.info("vector store function exited....")
            return vector_db
        except Exception as e:
            print(f"error occured in: {e}")
            logger.error(f"error occured in: {e}")   

    def initialized_retriver_fun(self) -> VectorStoreRetriever:

        try:
            chunking_fun = self.chunking_fun()
            embedding_fun = self.embedding_function()
            vectorstore = self.vector_store_fun(
                documents=chunking_fun,
                embedd_fun=embedding_fun
            )
            return vectorstore.as_retriever(
                search_kwargs={
                    "k":3
                }
            )
        except Exception as e:
            print(f"error occured in: {e}")
            logger.error(f"error occured in: {e}")        
            