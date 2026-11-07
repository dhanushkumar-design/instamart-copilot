
from src.agent import RefundAgent, SimulatedCrash
import pandas as pd

agent = RefundAgent()

scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
scenario = scenarios.iloc[0]

try:
    agent.run_agent(
        scenario.message,
        scenario.scenario_id,
        crash_after_tools=2
    )

except SimulatedCrash as error:
    print("CRASH RUN ID:", error.run_id)

    result = agent.resume_agent(error.run_id)

    print("STATUS:", result["status"])
    print("TOOLS:", [step["tool"] for step in result["steps"]])
    print("LLM CALLS:", result["llm_calls"])
    print("ANSWER:", result["final_answer"])
