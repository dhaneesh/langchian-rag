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
  

def create_retriveal_chain_lcel():
  retrieval_chain = (
    RunnablePassthrough.assign(
      context=itemgetter("questions") | retriever | format_docs
      | prompt_template
      | llm
      | StrOutputParser()
    )
  )
  return retrieval_chian

if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "what is Pinecone in machine learning?"
  
    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)
  
    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)


    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
