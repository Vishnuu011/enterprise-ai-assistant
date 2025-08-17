from src.enterprise_ai_assistant.stategraph.nodes_and_workflow import CyclicGraphsWorkflow

import streamlit as st
import os

os.makedirs("csv_data", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("sql", exist_ok=True)

st.set_page_config(page_title="Enterprise Multi-Agent AI Assistant", layout="wide")

st.title("🤖 Enterprise Multi-Agent AI Assistant")

if "app" not in st.session_state:
    st.session_state.app = CyclicGraphsWorkflow().build_graph_workflow()

if "chat_history" not in st.session_state:
    st.session_state.chat_history =[]


query = st.text_input("💬 Ask your question:")
uploaded_csv = st.file_uploader("📂 Upload CSV for Dataframe Agent", type=["csv"])
uploaded_doc = st.file_uploader("📄 Upload Document for RAG", type=["pdf", "txt", "docx"])
uploaded_sql = st.file_uploader("🗄 Upload SQLite DB", type=["db", "sqlite"])


if uploaded_csv is not None:
    csv_path = os.path.join("csv_data", uploaded_csv.name)
    with open(csv_path, "wb") as f:
        f.write(uploaded_csv.getbuffer())
    st.success(f"✅ CSV saved to {csv_path}")


if uploaded_doc is not None:
    doc_path = os.path.join("data", uploaded_doc.name)
    with open(doc_path, "wb") as f:
        f.write(uploaded_doc.getbuffer())
    st.success(f"✅ Document saved to {doc_path}")


if uploaded_sql is not None:
    sql_path = os.path.join("sql", uploaded_sql.name)
    with open(sql_path, "wb") as f:
        f.write(uploaded_sql.getbuffer())
    st.success(f"✅ SQLite DB saved to {sql_path}")


if st.button("Ask") and query:
    with st.spinner("🤔 Thinking..."):
        # Prepare input state
        init_state = {
            "query": query,
            "chat_history": st.session_state.chat_history,
            "generation": "",
            "documents": [],
        }

        # Run graph (direct invoke, no streaming)
        final_state = st.session_state.app.invoke(init_state)

    # Save chat history
    answer = final_state.get("generation", "")
    st.session_state.chat_history.append(f"User: {query}")
    st.session_state.chat_history.append(f"Assistant: {answer}")

    # Show Answer
    st.success(answer)

# Show chat history
st.subheader("Chat History")
for msg in st.session_state.chat_history:
    if msg.startswith("User:"):
        st.markdown(f"**🧑 {msg[5:]}**")
    else:
        st.markdown(f"**🤖 {msg[10:]}**")



  



