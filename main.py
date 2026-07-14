from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_core.runnables import RunnablePassthrough,RunnableLambda

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory,BaseChatMessageHistory
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from pydantic import BaseModel,Field
from typing import Dict,List,Optional
from datetime import datetime

from dotenv import load_dotenv
import json

load_dotenv()

#----------------------------Data Models----------------------------
class ResearchResponse(BaseModel):
    answer : str = Field(description='The answer to the question is')
    confidence : str = Field(description="high,Med or Low")
    sources : List[str] = Field(description='List of source documents used')
    key_quotes : List[str] = Field(description='Relevent quotes from the sources',default=[])
    follow_up_question : List[str] = Field(description='Suggested follow_up questions') 

#----------------------------------------------------------
#------------------Research Assistent Class------------------------
#----------------------------------------------------------

class AIResearchAssistent:
    def __init__(
    self,
    persist_directory: str = "./research_db",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
    ):
        self.persist_directory = persist_directory

        #1. Embedding : Turns text into embeddings
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

        #2. Splitters : Turns big docs into chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators=['\n\n','\n','.',' ','']
        )

        #3. Vector Store : stores and searches embeddings
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name='research_docs'
        )

        print("Research Assistent initialized")
        print(f"Vector store : {persist_directory}")
        print(f"Document indexed : {self.vectorstore._collection.count()}")
        
    def add_documents(
            self,
            documents : List[Document],
            source_name : Optional[str] = None, 
    )->int:
        
        #tag with source name
        if source_name:
            for doc in documents:
                doc.metadata['source'] = source_name
        
        #Split into chunks
        chunks = self.splitter.split_documents(documents)

        #Timestamp each chunks
        for chunk in chunks:
            chunk.metadata['indexed_at'] = datetime.now().isoformat()
        
        #Store in vector DB
        self.vectorstore.add_documents(chunks)

        print(f"Added {len(chunks)} chunks from {len(documents)} document(s)")
        return len(chunks)

if __name__ == "__main__":
    assistent = AIResearchAssistent()
