import sys
from pathlib import Path
from typing import Dict

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import asyncio
import streamlit as st
from datetime import datetime
from src.core.services.chat_service import ChatService
from src.core.services.embedding import EmbeddingService
from src.config.settings import settings
from src.config.model_provider import ModelProvider
from src.utils.logging import logger
from src.core.services.db_service import DatabaseService

class StreamlitUI:
    def __init__(self):
        model_provider = ModelProvider()

        self.db_service = DatabaseService()
        self.embedding_service = EmbeddingService(model_provider.embed_model)
        self.chat_service = ChatService(
            model_provider.llm,
            self.db_service,
            self.embedding_service
        )

    async def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'db_service'):
            await self.db_service.close()

    def setup_page(self):
        st.title("Odoo Expert")
        st.write("Ask me anything about Odoo and I'll provide you with the best answers with references and citations!")

    def setup_sidebar(self):
        version_options = {
            "16.0": 160,
            "17.0": 170,
            "18.0": 180
        }
        selected_version = st.sidebar.selectbox(
            "Select Odoo Version",
            options=list(version_options.keys()),
            format_func=lambda x: f"Version {x}",
            index=2  # Default to 18.0
        )
        return version_options[selected_version]

    @staticmethod
    def display_chat_message(role: str, content: str):
        with st.chat_message(role):
            st.markdown(content)

    async def process_query(self, query: str, version: int):
        """Process a query and display the response."""
        try:
            # Show a loading message
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response_placeholder.markdown("Searching documentation...")

            # Get relevant chunks
            chunks = await self.chat_service.retrieve_relevant_chunks(query, version)
            
            if not chunks:
                with st.chat_message("assistant"):
                    st.error("No relevant documentation found for your query. Try rephrasing your question or choosing a different Odoo version.")
                return
            
            # Show processing message
            response_placeholder.markdown("Generating response...")
            
            # Prepare context and generate response
            context, sources = self.chat_service.prepare_context(chunks)
            
            full_response = ""
            try:
                response = await self.chat_service.generate_response(
                    query=query,
                    context=context,
                    conversation_history=st.session_state.conversation_history,
                    stream=True
                )

                async for chunk in response:
                    full_response += chunk.delta
                    response_placeholder.markdown(chunk.message.content)
                response_placeholder.markdown(full_response)
                    
                if full_response:
                    # Add to conversation history only if we got a valid response
                    st.session_state.conversation_history.append({
                        "user": query,
                        "assistant": full_response,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    response_placeholder.markdown("I couldn't generate a response. Please try rephrasing your question.")
                    
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                import traceback
                logger.error(traceback.format_exc())  # This will give you a full stack trace
                response_placeholder.markdown(f"Sorry, I encountered an error: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            import traceback
            logger.error(traceback.format_exc())  # This will give you a full stack trace
            with st.chat_message("assistant"):
                st.error(f"An error occurred while processing your query: {str(e)}")

    async def main(self):
        try:
            self.setup_page()
            version = self.setup_sidebar()

            if 'conversation_history' not in st.session_state:
                st.session_state.conversation_history = []

            for message in st.session_state.conversation_history:
                self.display_chat_message("user", message["user"])
                self.display_chat_message("assistant", message["assistant"])

            user_input = st.chat_input("Ask a question about Odoo...")

            if user_input:
                self.display_chat_message("user", user_input)
                await self.process_query(user_input, version)

            if st.button("Clear Conversation"):
                st.session_state.conversation_history = []
                st.rerun()
        finally:
            await self.cleanup()

def run_app():
    ui = StreamlitUI()
    asyncio.run(ui.main())

if __name__ == "__main__":
    run_app()