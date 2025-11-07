import os
import json
import streamlit as st
from dotenv import load_dotenv
from config import AGENT_PERSONAS, AGENT_PREFIX, AGENT_URLS, TOP_K
from chroma_utils import build_or_load_chroma
from memory import load_memory_file, append_memory, save_memory_file
from ui_helpers import format_history_for_prompt
from langchain_groq import ChatGroq

load_dotenv()
st.set_page_config(page_title="Multi-Agent RAG (streaming) with Memory", layout="wide")

# ---------------- Session Init ----------------
if "db_cache" not in st.session_state:
    st.session_state.db_cache = {}

if "memory" not in st.session_state:
    st.session_state.memory = load_memory_file()

# ---------------- UI ----------------
st.title("Multi-Agent <Ask right question to right agent>")

agent = st.selectbox("Select Agent", list(AGENT_URLS.keys()))
question = st.text_input("Ask your question")


if st.button("Send",width=300):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            if agent not in st.session_state.memory:
                st.session_state.memory[agent] = []
                save_memory_file(st.session_state.memory)

            append_memory(st.session_state.memory, agent, "user", question)

            if agent not in st.session_state.db_cache:
                with st.spinner(f"Preparing DB for {agent}, Maybe take some seconds first time"):
                    st.session_state.db_cache[agent] = build_or_load_chroma(agent, AGENT_URLS[agent])

            db = st.session_state.db_cache[agent]
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

            try:
                docs = retriever.invoke(question)
                if not docs or len(docs) == 0:
                    output_box = st.empty()
                    output_box.markdown("⚠️ I don't have enough knowledge to answer that based on my sources.")
                    st.stop()

                if len(docs) < 2:
                    output_box = st.empty()
                    output_box.markdown("⚠️ Low context retrieved, Maybe the question not in my knowledge.")
                    st.stop()

            except Exception as e:
                st.error(f"Retriever error: {e}")
                st.stop()

            context_text = "\n\n".join(d.page_content for d in docs)
            history_text = format_history_for_prompt(st.session_state.memory, agent)
            prompt = (
                f"{AGENT_PERSONAS.get(agent, '')}\n\n"
                f"Use the context and chat history only to answer.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Chat history:\n{history_text}\n\n"
                f"User question:\n{question}\n\n"
                "Answer:"
            )

            api_key = os.environ.get("groq_key")
            if not api_key:
                st.error("Missing 'groq_key' env var.")
                st.stop()

            llm = ChatGroq(model=os.environ.get("GROQ_MODEL","llama-3.3-70b-versatile"), temperature=0, api_key=api_key)
            output_box = st.empty()
            full_answer = ""
            try:
                for chunk in llm.stream(prompt):
                    piece = chunk.content if hasattr(chunk, "content") else str(chunk)
                    full_answer += piece
                    output_box.markdown(full_answer)
            except Exception as e:
                st.error(f"Streaming error: {e}")
            else:
                final_answer = AGENT_PREFIX.get(agent, "") + full_answer

                append_memory(st.session_state.memory, agent, "assistant", final_answer)

                output_box.markdown(final_answer)
                st.success("Done ✅")




col1, col2, col3 = st.columns([1,1,2])

with col2:
    if st.button("Clear memory for this agent"):
        st.session_state.memory[agent] = []
        save_memory_file(st.session_state.memory)
        st.success(f"Cleared memory for {agent}")

with col3:
    if st.button("Export memory JSON"):
        st.download_button(
            label="Download memory.json",
            data=json.dumps(st.session_state.memory, ensure_ascii=False, indent=2),
            file_name="agent_memory.json",
            mime="application/json"
        )



# ---------------- Show conversation ----------------
# st.markdown("---")
# st.subheader("Conversation history (this agent)")
# hist = st.session_state.memory.get(agent, [])
# if not hist:
#     st.info("Start yor conversation.")
# else:
#     for m in hist:
#         role = m.get("role", "user")
#         text = m.get("text", "")
#         st.markdown(f"**{role.capitalize()}:** {text}")
