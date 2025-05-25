import os
import re
from openai import OpenAI
from dotenv import load_dotenv

def call_llm(prompt):    
    load_dotenv(override=True)
    client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), 
                    base_url=os.environ.get("DEEPSEEK_API_BASE_URL"))
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content

def extract_final_answer(solution):
    """
    Extract the final answer from a solution using regex patterns.

    Args:
        solution: The solution text

    Returns:
        Optional[str]: The extracted answer, or None if no answer pattern is found
    """
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

    # If no pattern matched, try to find the last number in the solution
    numbers = re.findall(r"[0-9.]+", solution)
    if numbers:
        return numbers[-1]

    return None

# Example usage
if __name__ == "__main__":
    print(call_llm("Tell me a short joke")) 