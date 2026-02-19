import os
import operator from itemgetter

from dotenv import load_dotenv
from langchain_core.message import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectoreStore


load_dotenv()
print("Initializing components...")
embedddings = OpenAIEmbeddings()
llm = ChatOpenAI()
vectorstore = PineconeVectorStore(
  index_name=os.enviorn["INDEX_NAME"], embedding=embedddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
prompt_template = ChatPromptTemplate.from_template(
  """Answer the question based only on the following context:

  {context}

  Question: {question}

  Provide a detailed answer:"""
)

def format_docs(docs):
  """Format retrieved documents into a single string."""
  return "\n\n".join(doc.page_conetent for doc in docs)

# ============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# ============================================================================


def retrieveal_chain_witout_lcel(query:str):
  """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
  
  #Step 1: Retrieve relevant documents
  docs = retriever.invoke(query)
  #Step 2: Format documents into context string
  context = format_docs(docs)
  #Step 3: Format the prompt with context and question
  messages = prompt_template.format_message(context=context, question=query)
  #Step 4 : Invoke LLM with the formatted mesagges
  response = llm.invoke(messages)

  return response.content
  














# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain
