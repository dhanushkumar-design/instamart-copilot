
"""Part D: Offline ReAct refund agent with crash-resume support."""

import json
import sqlite3
import time
import uuid
from pathlib import Path

import pandas as pd
import mock_llm

from .foundation import BASE, DB, RAW
from .policy import PolicyAssistant


OUTPUTS = BASE / "outputs"
STATE = BASE / "state"

OUTPUTS.mkdir(parents=True, exist_ok=True)
STATE.mkdir(parents=True, exist_ok=True)

LEDGER_FILE = OUTPUTS / "refund_ledger.json"
RUNS_FILE = OUTPUTS / "agent_runs.jsonl"


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Find an order and check its details.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Search refund and delivery policies.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate refund based on delivery delay.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Simulate issuing an eligible refund up to INR 1000.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate eligible refunds above INR 1000.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]


def read_ledger():
    if LEDGER_FILE.exists():
        return json.loads(LEDGER_FILE.read_text(encoding="utf-8"))
    return []


def save_ledger(entries):
    LEDGER_FILE.write_text(
        json.dumps(entries, indent=2),
        encoding="utf-8"
    )


class SimulatedCrash(Exception):
    def __init__(self, run_id):
        self.run_id = run_id
        super().__init__(f"Simulated crash after checkpoint: {run_id}")


class RefundAgent:
    def __init__(self, max_steps=8, token_budget=6000):
        self.max_steps = max_steps
        self.token_budget = token_budget
        self.policy = PolicyAssistant()
        self.customer_id = None
        self.scenario_id = None
        self.run_id = None

        self.TOOL_MAP = {
            "get_order": self.get_order,
            "search_policy": self.search_policy,
            "calculate_refund": self.calculate_refund,
            "issue_refund": self.issue_refund,
            "escalate_to_human": self.escalate_to_human
        }

    def get_order(self, order_id):
        with sqlite3.connect(DB) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM orders WHERE order_id = ?",
                (order_id,)
            ).fetchone()

        if row is None:
            return {"error": "order not found"}

        order = {
            key: row[key]
            for key in [
                "order_id",
                "customer_id",
                "status",
                "order_amount",
                "delivery_minutes"
            ]
        }

        if (
            self.customer_id is not None
            and order["customer_id"] != self.customer_id
        ):
            return {"error": "order not found"}

        return order

    def search_policy(self, query):
        matches = self.policy.retrieve(query, k=3)

        return {
            "matches": [
                {
                    "source": source,
                    "score": score,
                    "text": chunk
                }
                for source, chunk, score in matches
            ]
        }

    def calculate_refund(self, order_amount, delivery_minutes):
        if order_amount is None or delivery_minutes is None:
            return {"refund_inr": 0.0}

        amount = (
            round(float(order_amount), 2)
            if float(delivery_minutes) > 45
            else 0.0
        )

        return {"refund_inr": amount}

    def _record_action(self, order_id, refund_inr, action):
        order = self.get_order(order_id)

        if "error" in order:
            return {"error": "order not found"}

        if order["status"] != "completed":
            return {"error": "only completed orders are eligible"}

        expected = self.calculate_refund(
            order["order_amount"],
            order["delivery_minutes"]
        )["refund_inr"]

        if expected <= 0:
            return {"error": "delivery is not eligible for refund"}

        if round(float(refund_inr), 2) != expected:
            return {"error": "refund amount mismatch"}

        if action == "issued" and expected > 1000:
            return {"error": "human approval required"}

        if action == "escalated" and expected <= 1000:
            return {"error": "human escalation not required"}

        ledger = read_ledger()

        # Repeated simulation must not issue another refund.
        for entry in ledger:
            if entry["order_id"] == order_id:
                if (
                    entry["action"] == action
                    and entry["refund_inr"] == expected
                ):
                    return {
                        "status": action,
                        "refund_inr": expected,
                        "already_recorded": True
                    }

                return {"error": "order already processed"}

        ledger.append({
            "order_id": order_id,
            "customer_id": order["customer_id"],
            "refund_inr": expected,
            "action": action,
            "scenario_id": self.scenario_id,
            "run_id": self.run_id
        })

        save_ledger(ledger)

        return {
            "status": action,
            "refund_inr": expected,
            "simulated": True
        }

    def issue_refund(self, order_id, refund_inr):
        return self._record_action(
            order_id, refund_inr, "issued"
        )

    def escalate_to_human(self, order_id, refund_inr):
        return self._record_action(
            order_id, refund_inr, "escalated"
        )

    def _scenario(self, scenario_id):
        scenarios = pd.read_csv(RAW / "agent_scenarios.csv")
        matches = scenarios[
            scenarios["scenario_id"] == scenario_id
        ]

        if matches.empty:
            raise ValueError(f"Unknown scenario: {scenario_id}")

        return matches.iloc[0]

    def _checkpoint(self, state):
        path = STATE / f"{state['run_id']}.json"
        path.write_text(
            json.dumps(state, indent=2),
            encoding="utf-8"
        )

    def _finish(self, state, status, answer):
        result = {
            "run_id": state["run_id"],
            "scenario_id": state["scenario_id"],
            "status": status,
            "final_answer": answer,
            "steps": state["steps"],
            "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"]
        }

        with RUNS_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result) + "\n")

        return result

    def _execute(
        self,
        state,
        max_iterations,
        token_limit,
        time_limit_seconds,
        crash_after_tools=None
    ):
        self.customer_id = state["customer_id"]
        self.scenario_id = state["scenario_id"]
        self.run_id = state["run_id"]

        start = time.monotonic()

        while state["llm_calls"] < max_iterations:
            if time.monotonic() - start >= time_limit_seconds:
                return self._finish(
                    state,
                    "FALLBACK_TIMEOUT",
                    "Timeout reached. Handing over to a human."
                )

            response = mock_llm.chat(
                state["messages"],
                tools=TOOLS_SPEC
            )

            state["llm_calls"] += 1
            state["tokens_used"] += response.usage.total_tokens

            # Check budget before executing requested tools.
            if state["tokens_used"] > token_limit:
                return self._finish(
                    state,
                    "FALLBACK_TOKEN_LIMIT",
                    "Token limit exceeded. Handing over to a human."
                )

            message = response.choices[0].message
            calls = message.tool_calls or []

            if not calls:
                answer = message.content or "No answer available."
                status = answer.split(":", 1)[0].strip()

                return self._finish(state, status, answer)

            serialized_calls = [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments
                }
                for call in calls
            ]

            state["messages"].append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": serialized_calls
            })

            for call in calls:
                try:
                    arguments = (
                        call.arguments
                        if isinstance(call.arguments, dict)
                        else json.loads(call.arguments)
                    )

                    tool_function = self.TOOL_MAP.get(call.name)

                    if tool_function is None:
                        observation = {"error": "unknown tool"}
                    else:
                        observation = tool_function(**arguments)

                except (ValueError, TypeError, KeyError) as error:
                    observation = {"error": str(error)}

                state["messages"].append({
                    "role": "tool",
                    "name": call.name,
                    "tool_call_id": call.id,
                    "content": json.dumps(observation)
                })

                state["steps"].append({
                    "step": len(state["steps"]) + 1,
                    "tool": call.name,
                    "arguments": arguments,
                    "observation": observation
                })

                state["executed_tools"].append(call.name)

                # Save immediately after every tool observation.
                self._checkpoint(state)

                if (
                    crash_after_tools is not None
                    and len(state["executed_tools"]) == crash_after_tools
                ):
                    raise SimulatedCrash(state["run_id"])

        return self._finish(
            state,
            "FALLBACK_MAX_ITERATIONS",
            "Maximum iterations reached. Handing over to a human."
        )

    def run_agent(
        self,
        user_message,
        scenario_id,
        max_iterations=8,
        token_limit=6000,
        time_limit_seconds=10,
        crash_after_tools=None
    ):
        scenario = self._scenario(scenario_id)
        run_id = uuid.uuid4().hex

        state = {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "customer_id": scenario["customer_id"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "AGENT. Use the available tools to handle "
                        "refund requests safely. Never make real payments."
                    )
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "tokens_used": 0,
            "llm_calls": 0,
            "step_number": 0,
            "steps": [],
            "executed_tools": []
        }

        return self._execute(
            state,
            max_iterations,
            token_limit,
            time_limit_seconds,
            crash_after_tools
        )

    def resume_agent(
        self,
        run_id,
        max_iterations=8,
        token_limit=6000,
        time_limit_seconds=10
    ):
        checkpoint = STATE / f"{run_id}.json"

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {run_id}"
            )

        state = json.loads(
            checkpoint.read_text(encoding="utf-8")
        )

        return self._execute(
            state,
            max_iterations,
            token_limit,
            time_limit_seconds
        )

    # Compatibility with earlier project code.
    def run(self, customer_id, message):
        scenarios = pd.read_csv(RAW / "agent_scenarios.csv")
        matches = scenarios[
            (scenarios["customer_id"] == customer_id)
            & (scenarios["message"] == message)
        ]

        if matches.empty:
            return {
                "outcome": "NEEDS_INFO",
                "text": "Scenario not found.",
                "steps": 0,
                "tokens": 0
            }

        scenario_id = matches.iloc[0]["scenario_id"]
        result = self.run_agent(message, scenario_id)

        outcome = result["status"]
        if outcome.startswith("FALLBACK_"):
            outcome = "FALLBACK"

        return {
            "outcome": outcome,
            "text": result["final_answer"],
            "steps": len(result["steps"]),
            "tokens": result["tokens_used"]
        }


def run_agent(
    user_message,
    scenario_id,
    max_iterations=8,
    token_limit=6000,
    time_limit_seconds=10
):
    return RefundAgent().run_agent(
        user_message,
        scenario_id,
        max_iterations,
        token_limit,
        time_limit_seconds
    )


def resume_agent(run_id):
    return RefundAgent().resume_agent(run_id)


def evaluate():
    agent = RefundAgent()
    scenarios = pd.read_csv(RAW / "agent_scenarios.csv")
    results = []

    for row in scenarios.itertuples(index=False):
        result = agent.run_agent(
            row.message,
            row.scenario_id
        )

        actual = result["status"]
        comparable = (
            "FALLBACK"
            if actual.startswith("FALLBACK_")
            else actual
        )

        results.append({
            "scenario_id": row.scenario_id,
            "expected": row.expected_outcome,
            "outcome": actual,
            "correct": comparable == row.expected_outcome,
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": len(result["steps"])
        })

    frame = pd.DataFrame(results)

    (BASE / "reports").mkdir(exist_ok=True)

    frame.to_csv(
        BASE / "reports/agent_eval_results.csv",
        index=False
    )

    return {
        "pass_rate": round(float(frame.correct.mean()), 4),
        "passed": int(frame.correct.sum()),
        "total": len(frame)
    }
