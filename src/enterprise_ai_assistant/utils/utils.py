import os, sys
import warnings 

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    UnstructuredFileLoader,
    TextLoader
)

from typing_extensions import (
    List,
    Literal
)

from langchain.schema import Document

import logging

warnings.filterwarnings("ignore")

logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)


class StructuredAndUnstructuredFileLoader:

    def __init__(self, file_path: str):

        self.file_path = file_path

    def file_loader_fun(self) -> List[Document]:

        try:
            logger.info("structured and unstructured loader is started..")

            loader = list()

            loader.append(
                DirectoryLoader(
                    path=self.file_path,
                    glob="**/*.pdf",
                    recursive=True,
                    loader_cls=PyPDFLoader,
                    show_progress=True
                )
            )

            loader.append(
                DirectoryLoader(
                    path=self.file_path,
                    glob="**/*.txt",
                    recursive=True,
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"},
                    show_progress=True
                )
            )

            for ext in ["docx", "html", "htm", "md"]:

                loader.append(
                    DirectoryLoader(
                        path=self.file_path,
                        glob=f"**/*.{ext}",
                        recursive=True,
                        loader_cls=UnstructuredFileLoader,
                        show_progress=True
                    )
                )

            all_docs = list()

            for doc_loader in loader:
                try:
                    doc = doc_loader.load()
                    all_docs.extend(doc)
                except Exception as e:
                    print(f"Error loading with {loader.loader_cls.__name__}: {e}") 

            logger.info("structured and unstructured loader is exit...") 
                   
            return all_docs           
        except Exception as e:
            print(f"error occured in: {e}")
            logger.error(f"error occured in: {e}")    