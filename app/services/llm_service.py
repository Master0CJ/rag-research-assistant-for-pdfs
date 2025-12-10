# from langchain_openai import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from config import Config
# import os  # NEW: Added for environment variable access

# # NEW: Optional import for explicit tracing control
# import langsmith as ls

# class LLMService:
#     def __init__(self, vector_store):
#         # UNCHANGED: Original LLM configuration
#         self.llm = ChatOpenAI(
#             temperature=0.7,
#             model_name="gpt-3.5-turbo",
#             openai_api_key=Config.OPENAI_API_KEY
#         )
        
#         # UNCHANGED: Original memory configuration
#         self.memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             return_messages=True
#         )
        
#         # MODIFIED: Added .with_config() method to enable enhanced tracing
#         self.chain = ConversationalRetrievalChain.from_llm(
#             llm=self.llm,
#             retriever=vector_store.vector_store.as_retriever(),
#             memory=self.memory
#         ).with_config({  # NEW: Configuration for LangSmith tracing
#             "tags": ["conversational-rag", "production"],  # NEW: Tags for trace organization
#             "metadata": {"service": "llm_service", "version": "1.0"}  # NEW: Metadata for debugging
#         })

#     # UNCHANGED: Original method (will now automatically trace when env vars are set)
#     def get_response(self, query):
#         try:
#             # The chain will automatically be traced when environment variables are set
#             response = self.chain.invoke({"question": query})
#             return response['answer']
#         except Exception as e:
#             print(f"Error getting LLM response: {e}")
#             return "I encountered an error processing your request."

#     
