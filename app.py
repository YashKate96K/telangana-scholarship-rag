import streamlit as st
import uuid

from rag_pipline import ask_scholarship_assistant

st.set_page_config(page_title="Telangana Scholarship Assistant", page_icon="🎓")
st.title("🎓 Telangana Scholarship Assistant")

if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.conversations[new_id] = {"title": "New chat", "display_messages": []}
    st.session_state.current_id = new_id

current = st.session_state.conversations[st.session_state.current_id]

with st.sidebar:
    if st.button("➕ New chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.conversations[new_id] = {"title": "New chat", "display_messages": []}
        st.session_state.current_id = new_id
        st.rerun()

    st.markdown("---")
    st.caption("Past conversations")
    for conv_id in reversed(list(st.session_state.conversations.keys())):
        conv = st.session_state.conversations[conv_id]
        is_active = conv_id == st.session_state.current_id
        label = ("🟢 " if is_active else "") + conv["title"]
        if st.button(label, key=f"conv_{conv_id}", use_container_width=True):
            st.session_state.current_id = conv_id
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Delete this conversation", use_container_width=True):
        del st.session_state.conversations[st.session_state.current_id]
        if not st.session_state.conversations:
            new_id = str(uuid.uuid4())
            st.session_state.conversations[new_id] = {"title": "New chat", "display_messages": []}
            st.session_state.current_id = new_id
        else:
            st.session_state.current_id = list(st.session_state.conversations.keys())[-1]
        st.rerun()

for msg in current["display_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question...")

if user_input:
    if not current["display_messages"]:
        current["title"] = user_input[:40] + ("..." if len(user_input) > 40 else "")

    with st.chat_message("user"):
        st.markdown(user_input)
    current["display_messages"].append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Looking it up..."):
            result = ask_scholarship_assistant(user_input)
            answer = result["answer"]
            st.markdown(answer)

    current["display_messages"].append({"role": "assistant", "content": answer})
    st.rerun()