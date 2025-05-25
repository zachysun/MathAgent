import re
import math
import numpy as np
from typing import List, Optional
from langchain_core.tools import BaseTool


class MathTools:
    """Collection of tools for the MathAgent."""

    @staticmethod
    def extract_final_answer(solution: str) -> Optional[str]:
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


class CalculateTool(BaseTool):
    """Tool for performing basic calculations."""

    name: str = "calculate"
    description: str = "Calculates the result of a mathematical expression."

    def _run(self, expression: str) -> str:
        try:
            safe_dict = {
                "abs": abs,
                "pow": pow,
                "round": round,
                "int": int,
                "float": float,
                "max": max,
                "min": min,
                "sum": sum,
                "len": len,
                "math": math,
                "np": np,
            }

            # Add all math module functions
            for name in dir(math):
                if not name.startswith("__"):
                    safe_dict[name] = getattr(math, name)

            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return str(result)
        except Exception as e:
            return f"Error in calculation: {str(e)}"


def get_math_tools() -> List[BaseTool]:
    """
    Get a list of all available math tools.

    Returns:
        List[BaseTool]: A list of math tools
    """
    return [
        CalculateTool(),
    ]
