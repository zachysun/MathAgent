import re
from typing import Optional


class MathTools:
    """Collection of mathematical utility functions for the Math reasoning workflow."""

    @staticmethod
    def extract_final_answer(solution: str) -> Optional[str]:
        """
        Extract the answer from a solution using regex patterns.

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


def get_calculator_tools():
    """
    Get Agno calculator tools.

    Returns:
        The calculator tools from Agno
    """
    from agno.tools.calculator import CalculatorTools

    return CalculatorTools(
        add=True,
        subtract=True,
        multiply=True,
        divide=True,
        exponentiate=True,
        factorial=True,
        is_prime=True,
        square_root=True,
    )
