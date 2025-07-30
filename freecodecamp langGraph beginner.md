
---

# 1. Agent_Bot.py (Simple Chat Agent)

### **What does it do?**
- It’s a basic chatbot: you type something, it replies using AI.
- No memory of past messages.

### **How does it work?**
- Takes your input.
- Calls the AI (GPT-4o) for a response.
- Prints the reply.

### **Code Example (with comments):**
```python
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv # for secret keys

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

# Define the AI model
llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

# Set up the graph flow: user input -> AI -> end
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# Main loop to chat
user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
```

### **Key idea:**  
One-shot chat—no memory, just back-and-forth.

---

# 2. Memory_Agent.py (Chat Agent with Memory)

### **What does it do?**
- Remembers the whole conversation so far.

### **How does it work?**
- Keeps a list of all messages.
- Passes this list to the AI for context.
- Writes a log of your chat at the end.

### **Code Example:**
```python
import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    print("CURRENT STATE: ", state["messages"])
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []
user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")

# Save the conversation to a file
with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    for message in conversation_history:
        file.write(str(message) + "\n")
```

### **Key idea:**  
Keeps chat history and passes it to the AI so it understands the context.

---

# 3. ReAct.py (Reasoning + Action Agent)

### **What does it do?**
- Can do calculations (add, subtract, multiply) and answer general questions.
- Decides whether to use a tool or just respond.

### **How does it work?**
- Has math tools.
- Checks if it needs to use a tool after each step (“should_continue” logic).
- Cycles between thinking and acting until finished.

### **Code Example:**
```python
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Math tools
@tool
def add(a: int, b:int):
    return a + b

@tool
def subtract(a: int, b: int):
    return a - b

@tool
def multiply(a: int, b: int):
    return a * b

tools = [add, subtract, multiply]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)
graph.set_entry_point("our_agent")
graph.add_conditional_edges("our_agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        print(message.content)

inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))
```

### **Key idea:**  
Can use tools and “think” in steps, not just reply directly.

---

# 4. Drafter.py (Document Editing Agent)

### **What does it do?**
- Helps you draft, update, and save documents.
- Uses tools for editing and saving.

### **How does it work?**
- Has tools for updating and saving.
- Asks you what you want to do.
- Shows the current document after changes.

### **Code Example:**
```python
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"

@tool
def save(filename: str) -> str:
    global document_content
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        return f"Document has been saved successfully to '{filename}'."
    except Exception as e:
        return f"Error saving document: {str(e)}"

tools = [update, save]
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    The current document content is:{document_content}
    """)
    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        user_message = HumanMessage(content=user_input)
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_messages)
    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    if not messages:
        return "continue"
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and "saved" in message.content.lower() and "document" in message.content.lower()):
            return "end"
    return "continue"

graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges("tools", should_continue, {"continue": "agent", "end": END})
app = graph.compile()
```

### **Key idea:**  
Has specialized tools for editing and saving documents, and tracks the document state.

---

# 5. RAG_Agent.py (Retrieval-Augmented Agent)

### **What does it do?**
- Answers questions using info from a specific PDF (e.g., stock market data).
- Uses retrieval, not just guessing.

### **How does it work?**
- Loads and splits the PDF.
- Embeds and stores the chunks in ChromaDB.
- Uses a retriever tool to find relevant info.
- Cites the parts of the document it uses.

### **Code Example:**
```python
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

pdf_path = "Stock_Market_Performance_2024.pdf"
if not os.path.exists(pdf_path):
    raise Exception("PDF not found!")

loader = PyPDFLoader(pdf_path)
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
docs = splitter.split_documents(pages)
persist_directory = "chroma_db"
collection_name = "stock_market"
vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory, collection_name=collection_name)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

@tool
def retriever_tool(query: str) -> str:
    docs = retriever.invoke(query)
    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    return "\n\n".join(results)

tools = [retriever_tool]
llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}

# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        if not t['name'] in tools_dict:
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    return {'messages': results}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")
rag_agent = graph.compile()

def running_agent():
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)
```

### **Key idea:**  
Retrieves *real* info from a document using embeddings and a database, then answers your questions.

---

# **Comparison Table**

| Agent Name      | Memory | Uses Tools | Specialized For | Code Pattern Highlights              |
|-----------------|--------|------------|-----------------|--------------------------------------|
| Agent_Bot       | No     | No         | Simple chat     | Single process node                  |
| Memory_Agent    | Yes    | No         | Ongoing chat    | Stores and passes message history    |
| ReAct           | No     | Yes        | Reasoning/tasks | Conditional step, tool use, looping  |
| Drafter         | No     | Yes        | Document editing| Specialized tools for doc editing    |
| RAG_Agent       | Partial| Yes        | Fact lookup     | Uses retrieval, database, citations  |

---

# **How to Learn Fast from These Agents**

- **Run each one and follow the code logic:** See how `AgentState` is built and passed around.
- **Notice the use of tools:** Functions decorated with `@tool` get “called” by the AI to do jobs.
- **Graphs = workflows:** Each agent is a graph of nodes (steps) and edges (transitions).
- **Memory vs. No Memory:** If you want the AI to remember, keep a message list. If not, single message each time.

---

