from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory  # ✅ always available
from langchain_core.runnables.history import RunnableWithMessageHistory
import os, dotenv

dotenv.load_dotenv()
llm = ChatOpenAI(model_name="gpt-4o-mini",
                 max_tokens=100,
                 temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful history professor named Herodotus Junior."),
        MessagesPlaceholder("history"),
        ("human", "{user_input}"),
    ]
)

pipeline = prompt | llm      # any LCEL graph works here

store = {}                   # one history object per session
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chat = RunnableWithMessageHistory(
    runnable=pipeline,
    get_session_history=get_session_history,
    input_messages_key="user_input",
    history_messages_key="history",
)

SESSION = {"configurable": {"session_id": "demo-1"}}

# print(chat.invoke({"user_input": "When was Marcus Aurelius emperor?"}, config=SESSION).content)
print(chat.invoke({"user_input": "Who was he married to?"}, config=SESSION).content)
# print(chat.invoke({"user_input": "What? So Commodus existed? Like the movie?"}, config=SESSION).content)