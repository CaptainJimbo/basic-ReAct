# langgraph_beach_agent.py
"""
ReAct-style beach recommender built with LangGraph + LangChain tools.
Run:  python langgraph_beach_agent.py
"""

import os
import datetime
import random
from dotenv import load_dotenv
from typing import TypedDict

from langgraph.graph import StateGraph
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# ------------------------------------------------------------------ #
# 1.  ENVIRONMENT
# ------------------------------------------------------------------ #
load_dotenv()                           # expects OPENAI_API_KEY in .env

# ------------------------------------------------------------------ #
# 2.  STATE
#    • `total=False` → keys are optional; lets us add them gradually.
# ------------------------------------------------------------------ #
class AgentState(TypedDict, total=False):
    input: str          # user’s question
    final_answer: str   # agent’s answer at the end

# ------------------------------------------------------------------ #
# 3.  TOOLS
# ------------------------------------------------------------------ #
@tool
def get_location(_: str = "") -> str:
    """Return the user's current location."""
    return "Naousa, Paros, Greece"


@tool
def get_time(_: str = "") -> str:
    """Return current date-time (ISO)."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_weather(location: str) -> str:
    """Fake weather (wind dir/speed, waves, temp)."""
    wind_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    data = {
        "wind_direction": random.choice(wind_directions),
        "wind_speed_kmh": random.randint(10, 30),
        "wave_height_m": round(random.uniform(0.2, 1.5), 1),
        "temperature_c": random.randint(26, 32),
    }
    return (
        f"Wind: {data['wind_speed_kmh']} km/h from {data['wind_direction']}, "
        f"Waves: {data['wave_height_m']} m, Temp: {data['temperature_c']} °C"
    )


@tool
def get_traffic(location_pair: str) -> str:
    """Return fake drive-time between origin and destination."""
    try:
        origin, destination = [s.strip() for s in location_pair.split("to")]
    except ValueError:
        return "Please specify in the form 'Naousa to Kolymbithres'."
    minutes = random.randint(5, 25)
    return f"Estimated drive time from {origin} to {destination}: {minutes} minutes"


@tool
def list_beaches(_: str = "") -> str:
    """List nearby beaches with orientation."""
    beaches = [
        {"name": "Kolymbithres", "orientation": "North-West"},
        {"name": "Monastiri", "orientation": "North"},
        {"name": "Santa Maria", "orientation": "East"},
        {"name": "Golden Beach", "orientation": "South-East"},
    ]
    return "\n".join(f"{b['name']} – facing {b['orientation']}" for b in beaches)

TOOLS = [get_location, get_time, get_weather, get_traffic, list_beaches]

# ------------------------------------------------------------------ #
# 4.  LLM + ReAct agent
# ------------------------------------------------------------------ #
llm = ChatOpenAI(model="gpt-4o", temperature=0)
react_agent = create_react_agent(model=llm, tools=TOOLS)   # a Runnable

# ------------------------------------------------------------------ #
# 5.  GRAPH NODE: wraps `react_agent.invoke`
# ------------------------------------------------------------------ #
def agent_node(state: AgentState) -> AgentState:
    """Run one ReAct step and write the answer to state."""
    # create_react_agent accepts a *string* prompt
    result = react_agent.invoke(state["input"])

    # create_react_agent returns either a string or {"output": …}
    answer = result["output"] if isinstance(result, dict) else result

    # Return *only* the fields we’re updating (diff)
    return {"final_answer": answer}


# ------------------------------------------------------------------ #
# 6.  BUILD LangGraph
# ------------------------------------------------------------------ #
workflow = StateGraph(AgentState)
workflow.add_node("agent_step", agent_node)
workflow.set_entry_point("agent_step")
workflow.set_finish_point("agent_step")
graph = workflow.compile()

# ------------------------------------------------------------------ #
# 7.  DEMO
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    question = (
        "Where should I go swimming today if I want calm water and low wind? "
        "I also don't want to drive more than 15 minutes from my location."
    )

    # Initial state just needs the user input
    final_state = graph.invoke({"input": question})
    print("\n✅ Recommendation:", final_state["final_answer"])
