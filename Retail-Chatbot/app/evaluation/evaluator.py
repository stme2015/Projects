# Agent output scoring
from evaluation.test_cases import test_cases
from agents.manager import build_agent
from langchain_core.messages import HumanMessage

from utils.logging import log_info
from langsmith import trace

agent = build_agent()

@trace
def evaluate_graph(graph, input_data):
    return graph.invoke(input_data)

def evaluate_agent():
    results = []
    for case in test_cases:
        input_text = case["input"]
        expected = case["expected_keywords"]
        response = agent.run(input_text)
        score = all(keyword in response for keyword in expected)
        status = "PASS" if score else "FAIL"
        log_info(f"Evaluation {status}: {input_text}")
        results.append({
            "input": input_text,
            "response": response,
            "pass": score
        })
    return results

if __name__ == "__main__":
    eval_results = evaluate_agent()
    for r in eval_results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"Input: {r['input']}\nOutput: {r['response']}\nResult: {status}\n")
