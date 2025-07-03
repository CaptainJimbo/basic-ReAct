import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

_ = load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv('OPENAI_API_KEY')
)

# Tool for getting book genre
class GetBookGenreTool(BaseTool):
    name: str = "get_book_genre"
    description: str = "Returns the genre(s) of the given book. Input should be a book title."
    
    def _run(self, book: str) -> str:
        system_prompt = "Return the literary genre(s) of the book mentioned."
        user_prompt = f"The book:\n{book}"
        
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        return response.content.strip()

# Tool for getting similar books
class GetSimilarBooksTool(BaseTool):
    name: str = "get_similar_books"
    description: str = "Returns a list of 3-5 books similar to the one given. Input should be a book title."
    
    def _run(self, book: str) -> str:
        system_prompt = "Suggest 3–5 books that are similar in theme, style, or genre to the given book."
        user_prompt = f"The book:\n{book}"
        
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        return response.content.strip()

# Tool for getting books by author
class GetBooksByAuthorTool(BaseTool):
    name: str = "get_books_by_author"
    description: str = "Returns a list of notable books written by the given author. Input should be an author name."
    
    def _run(self, author: str) -> str:
        system_prompt = "List 3–5 notable books written by the given author."
        user_prompt = f"The author:\n{author}"
        
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        return response.content.strip()

# Create tools list
tools = [
    GetBookGenreTool(),
    GetSimilarBooksTool(),
    GetBooksByAuthorTool()
]

# Create ReAct prompt template
react_prompt = PromptTemplate.from_template("""
                                            
You are a helpful assistant that can help users with book recommendations.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
                                            
Thought: you should always think about what to do
                                            
Action: the action to take, should be one of [{tool_names}]
                                            
Action Input: the input to the action
  
Observation: the result of the action                                            
... (this Thought/Action/Action Input/Observation can repeat 5 times)
                                            
Thought: I now know the final answer
                                            
Final Answer: the final answer to the original input question

Begin!

Question: {input}
                                            
Thought: {agent_scratchpad}
""")


# Create the ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)
"""
* **Thought** (yellow) – the model’s internal chain-of-thought.
* **Action** (bright green) – the name of the tool the agent is about to call.
* **Action Input** (cyan) – the argument(s) passed to that tool.
* **Observation** (magenta) – the value returned by the tool.
* **Final Answer** (bold white) – the answer ultimately given back to the user.
"""

def query(question: str) -> str:
    """Query the ReAct agent with a question."""
    try:
        result = agent_executor.invoke({"input": question})
        return result["output"]
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":

    question = "I loved watching the Sandman on Netflix. It is based on books. What genre is it, and can you suggest similar books? Also, what other books did its author write?"
    print("Question:", question)
    print("-" * 88)
    result = query(question)
    print(result)
