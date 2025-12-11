from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from config import Config
import langsmith as ls

class EducationalLLMService:
    def __init__(self, vector_store):
        # Educational-specific prompt template
        self.educational_prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""
            You are an educational assistant helping students learn. Use the provided course materials to answer questions.
            
            Course Materials Context: {context}
            
            Chat History: {chat_history}
            
            Student Question: {question}
            
            Instructions:
            1. Provide clear, educational explanations
            2. Break down complex concepts into simple terms
            3. Include examples when helpful
            4. Cite specific sections from the course materials
            5. If the answer isn't in the materials, suggest related concepts that are covered
            6. Encourage further learning with follow-up questions
            
            Educational Response:
            """
        )
        
        self.llm = ChatOpenAI(
            temperature=0.3,  # Lower temperature for more consistent educational responses
            model="gpt-4",    # Use GPT-4 for better educational explanations
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.educational_prompt}
        ).with_config({
            "tags": ["educational-rag", "course-qa"],
            "metadata": {"service": "educational_assistant", "version": "1.0"}
        })

    def get_course_answer(self, question, subject="general", difficulty_level="undergraduate"):
        try:
            # Add educational context to the query
            enhanced_question = f"[Subject: {subject}] [Level: {difficulty_level}] {question}"
            
            with ls.tracing_context(project_name="educational-assistant", enabled=True):
                response = self.chain.invoke({"question": enhanced_question})
                return {
                    "answer": response['answer'],
                    "sources": self._extract_sources(response),
                    "subject": subject,
                    "difficulty": difficulty_level
                }
        except Exception as e:
            print(f"Error in educational response: {e}")
            return {
                "answer": "I'm having trouble accessing the course materials right now. Please try again.",
                "sources": [],
                "subject": subject,
                "difficulty": difficulty_level
            }
    
    def _extract_sources(self, response):
        # Extract source documents for citation
        sources = []
        if 'source_documents' in response:
            for doc in response['source_documents']:
                sources.append({
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                })
        return sources
