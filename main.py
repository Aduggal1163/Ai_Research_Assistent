from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
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
#----------------------------------------------------------
#----------------------------Data Models----------------------------
#----------------------------------------------------------
class ResearchResponse(BaseModel):
    answer : str = Field(description='The answer to the question is')
    confidence : str = Field(description="high,Med or Low")
    sources : List[str] = Field(description='List of source documents used')
    key_quotes : List[str] = Field(description='Relevent quotes from the sources',default=[])
    follow_up_question : List[str] = Field(description='Suggested follow_up questions') 

#----------------------------------------------------------
#------------------Research Assistent Class------------------------
#----------------------------------------------------------

class AIResearchAssistant:
    def __init__(
    self,
    persist_directory: str = "./research_db",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
    ):
        self.persist_directory = persist_directory

        #1. Embedding : Turns text into embeddings
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

        #3. Splitters : Turns big docs into chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators=['\n\n','\n','.',' ','']
        )
        #2 Llm
        self.llm = init_chat_model('gpt-4o-mini',temperature = 0)

    
        #4. Vector Store : stores and searches embeddings
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name='research_docs'
        )

        #5. Session store
        self.session_store : Dict[str,InMemoryChatMessageHistory] = {}


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

    def add_text(self, text: str, source: str, metadata: dict = None)->int:

        """Add single text string as a document"""
        doc = Document(
            page_content=text,
            metadata={
                'source':source,**(metadata or {})
            }
        )
        return self.add_documents([doc])
    
    def add_texts(self, texts: List[str], source: str) -> int:
        """Add multiple text strings from the same source."""
        docs = [Document(page_content=t, metadata={"source": source}) for t in texts]
        return self.add_documents(docs)
    
    def get_document_count(self)->int:
        """Get total number of indexed chunks"""
        return self.vectorstore._collection.count()
    
    def list_sources(self)->list[str]:
        """List all unique sources in the database"""
        result = self.vectorstore._collection.get()
        sources = set()
        for metadata in result.get('metadatas',[]):
            if metadata and 'source' in metadata:
                sources.add(metadata['source'])
        return sorted(list(sources))

    def _build_retriever(self):
        """Build a basic similarity retriever"""
        return self.vectorstore.as_retriever(
            search_type='similarity',
            search_kwargs={'k':4}
        )
    
    def format_docs_for_context(self,docs)->str:
        """Format retrieved documents into a string for the prompt."""
        if not docs:
            return "No documents found"
        formatted=[]
        for i,doc in enumerate(docs,start=1):
            source = doc.metadata.get('source','unknown')
            formatted.append(f'for chunk {i} and source [{source}]\n content is {doc.page_content}')
        return '\n\n--\n\n'.join(formatted)

    def get_session_history(self,session_id: str)->BaseChatMessageHistory:
        """Get or create session history"""
        if session_id not in self.session_store:
            self.session_store[session_id] = InMemoryChatMessageHistory()
        return self.session_store[session_id]

    def ask(
            self,
            question: str,
    )->str:
        """Ask a question and get a structured response"""
        retriever = self._build_retriever()
        docs = retriever.invoke(question)
        context = self.format_docs_for_context(docs)
        history = self.get_session_history('session_id')
        prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
                    """You are an AI Research Assistant. Analyze the provided documents 
    and return a structured response.

    Rules:
    1. ONLY use information from the provided context
    2. If the context doesn't have the answer, say so in the answer field
    3. Cite which sources are used.
    4. Rate your confidence High,Med or Low
    Use conversation history to understand follow-up questions.
    """
        ),
        MessagesPlaceholder(variable_name='history'),
        (
            "human",
            """Context documents:

{context}

Available sources: {sources}

Question: {question}"""
        ),
    ]
)
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke(
            {
            'context':context,
            'question':question,
            'sources': self.list_sources(),
            'history':history.messages[-10:]
            }
        )
    #Save to memory
        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=response))
        return response
    
    def clear_session(self, session_id: str):
        if session_id in self.session_store:
            self.session_store[session_id].clear()
            print(f"Cleared session: {session_id}")

    def get_session_messages(self, session_id: str) -> list:
        """Get conversation history as readable dicts."""
        if session_id not in self.session_store:
            return []
        return [
            {
                "role": "human" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in self.session_store[session_id].messages
        ]
    
    

if __name__ == "__main__":
    import shutil

    shutil.rmtree("./research_db", ignore_errors=True)
    assistant = AIResearchAssistant()

    # Add research docs
    assistant.add_text(
        """
        Attention Mechanisms in Neural Networks

        The attention mechanism was introduced in "Attention Is All You Need"
        by Vaswani et al. (2017). It allows models to focus on relevant parts
        of the input when generating output.

        Key concepts:
        - Query, Key, Value (QKV) triplets
        - Scaled dot-product attention
        - Multi-head attention for parallel processing

        The transformer architecture has become the foundation for modern NLP
        models including BERT, GPT, and T5.
        """,
        source="attention_mechanisms.pdf",
    )

    assistant.add_text(
        """
        Retrieval-Augmented Generation (RAG)

        RAG combines retrieval systems with generative models. First introduced
        by Lewis et al. (2020), RAG addresses the limitation of LLMs being
        limited to their training data.

        Components of a RAG system:
        1. Document store with vector embeddings
        2. Retriever to find relevant documents
        3. Generator (LLM) to produce responses

        Benefits include reduced hallucination, up-to-date information,
        and source attribution.
        """,
        source="rag_survey.pdf",
    )

    assistant.add_text(
        """
        LangChain and LangGraph Framework Overview

        LangChain is an open-source framework for building LLM applications.
        Key features include modular components, integration with 50+ LLM
        providers, and built-in RAG utilities.

        LangGraph extends LangChain for stateful applications with
        graph-based state management, support for cycles and loops,
        and human-in-the-loop workflows.
        """,
        source="langchain_docs.md",
    )

    print(f'\n Total documents in database are : {assistant.get_document_count()}')
    print(f"\n No of sources we have in our database are: {assistant.list_sources()}")


    # retriever = assistant._build_retriever()
    # response = retriever.invoke("What is RAG")
    ## for i,response in enumerate(response,start=1):
    ##    print(f"For chunk {i} \n sources are [{response.metadata['source']}]\n response is {response.page_content} ") 
    # print(assistant.format_docs_for_context(response))

    q1='what is rag and what are its main key components'
    print(f"\n User: {q1}")
    print(f"AI: {assistant.ask(q1)}")
    q2='what is attention mechanism in rag'
    print(f"\n User: {q2}")
    print(f"AI: {assistant.ask(q2)}")
    q3='Can you expand the second component you just explains'
    print(f"\n User: {q3}")
    print(f"AI: {assistant.ask(q3)}")
    # Cleanup
    shutil.rmtree("./research_db", ignore_errors=True)