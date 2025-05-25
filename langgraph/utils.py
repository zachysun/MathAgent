import os
import re
from openai import OpenAI
from dotenv import load_dotenv

def call_llm(prompt: str) -> str:
    load_dotenv(override=True)
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url=os.environ.get("DEEPSEEK_API_BASE_URL")
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def extract_final_answer(solution: str) -> str:
    """Extract final answer from LLM response"""
    answer_patterns = [
        r"Therefore,?\s*(?:the\s+)?answer\s+is\s*([0-9.]+)",
        r"The\s+final\s+answer\s+is\s*([0-9.]+)",
        r"Answer:\s*([0-9.]+)",
        r"Final\s+answer:\s*([0-9.]+)",
        r"Hence,?\s*(?:the\s+)?answer\s+is\s*([0-9.]+)",
        r"Thus,?\s*(?:the\s+)?answer\s+is\s*([0-9.]+)",
        r"\\boxed{([0-9.]+)}",
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, solution, re.IGNORECASE)
        if match:
            return match.group(1)

    # Fallback: find last number in response
    numbers = re.findall(r"[0-9.]+", solution)
    return numbers[-1] if numbers else solution

if __name__ == "__main__":
    print(call_llm("Tell me a short joke")) 