import os
import datetime
import random
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_openai import ChatOpenAI

# Load OpenAI key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Simulated Tool: Get current location
def get_location(_: str = "") -> str:
    return "Naousa, Paros, Greece"

# Simulated Tool: Get current time
def get_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Simulated Tool: Get weather data
def get_weather(location: str) -> str:
    wind_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    data = {
        "wind_direction": random.choice(wind_directions),
        "wind_speed_kmh": random.randint(10, 30),
        "wave_height_m": round(random.uniform(0.2, 1.5), 1),
        "temperature_c": random.randint(26, 32)
    }
    return f"Wind: {data['wind_speed_kmh']} km/h from {data['wind_direction']}, Waves: {data['wave_height_m']} m, Temp: {data['temperature_c']}°C"

# Simulated Tool: Get traffic time
def get_traffic(location_pair: str) -> str:
    try:
        origin, destination = [s.strip() for s in location_pair.split("to")]
    except:
        return "Please specify in the form 'Naousa to Kolymbithres'"
    
    minutes = random.randint(5, 25)
    return f"Estimated drive time from {origin} to {destination}: {minutes} minutes"


# Simulated Tool: List beaches with orientations
def list_beaches(location: str) -> str:
    beaches = [
        {"name": "Kolymbithres", "orientation": "Northwest"},
        {"name": "Monastiri", "orientation": "North"},
        {"name": "Santa Maria", "orientation": "East"},
        {"name": "Golden Beach", "orientation": "Southeast"},
    ]
    return "\n".join([f"{b['name']} - facing {b['orientation']}" for b in beaches])

# Wrap all tools
tools = [
    Tool(name="get_location", func=get_location, description="Get the user's current location"),
    Tool(name="get_time", func=get_time, description="Get current date and time"),
    Tool(name="get_weather", func=get_weather, description="Get weather info including wind direction and waves"),
    Tool(name="get_traffic", func=get_traffic, description="Estimate drive time between two places. Input format: 'Naousa to Kolymbithres'"),
    Tool(name="list_beaches", func=list_beaches, description="List beaches near a given location"),
]

# Use GPT-4o as the main agent LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o")

# Create the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run a sample query
query = "Where should I go swimming today if I want calm water and low wind? I also don't want to drive more than 15 minutes from my location."
result = agent.invoke(query)
print("\n✅ Recommendation:", result)
