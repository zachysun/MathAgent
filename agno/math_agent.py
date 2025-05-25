"""
Math reasoning workflow based on Agno framework.
"""

import os
import re
from typing import Iterator, Optional, List
from dotenv import load_dotenv
from pydantic import BaseModel
from agno.agent import Agent, RunResponse
from agno.models.deepseek import DeepSeek
from agno.workflow import RunEvent, RunResponse, Workflow
from utils import MathTools, get_calculator_tools
from prompts import MathPromptTemplates, get_critic_prompt


class MathWorkflowState(BaseModel):
    problem: str
    answer: Optional[str] = None
    answer_list: List[str] = []


class MathReasoningWorkflow(Workflow):
    """
    A math reasoning workflow with multiple reasoning agents.
    """

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        temperature: float = 1.0,
        token_limit: int = 8192,
        selection_strategy: str = "agent",
        **kwargs,
    ):
        """
        Initialize the MathReasoningWorkflow.

        Args:
            model_name: The name of the model to use (default: "deepseek-chat")
            api_key: API key for model provider (will override .env if provided)
            api_base_url: Base URL for API (will override .env if provided)
            temperature: Temperature for generation (default: 1.0)
            token_limit: Maximum token limit (default: 8192)
            selection_strategy: Strategy for selecting the best solution ("agent", "last", "all")
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        
        # Load environment variables from .env file if exists
        load_dotenv(override=True)

        # Initialize API key and base URL variables
        self.api_key = (api_key if api_key else os.environ.get("DEEPSEEK_API_KEY"))
        self.api_base_url = (api_base_url if api_base_url else os.environ.get("DEEPSEEK_API_BASE_URL"))

        self.model_name = model_name
        self.temperature = temperature
        self.token_limit = token_limit
        self.selection_strategy = selection_strategy

        # Initialize the reasoning agent
        self._initialize_agent()

    def _initialize_agent(self) -> None:
        """Initialize the reasoning agent with model and tools."""
        model_params = {
            "api_key": self.api_key,
            "base_url": self.api_base_url,
            "id": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.token_limit,
        }

        model = DeepSeek(**model_params)
        calculator_tools = get_calculator_tools()

        # Create the reasoning agent
        self.reasoner = Agent(
            model=model,
            introduction=MathPromptTemplates.SYSTEM_PROMPT,
            tools=[calculator_tools],
            show_tool_calls=True,
            markdown=True,
        )

    def run(self, problem: str, num_rounds: int = 3) -> Iterator[RunResponse]:
        """
        Execute the math reasoning workflow.
        """
        # Initialize state
        state = MathWorkflowState(problem=problem)
        self.session_state[problem] = state

        # Set state
        state.answer_list = []
        state.answer = None
        self.session_state[problem] = state

        # Multi-round reasoning
        for round_idx in range(num_rounds):
            if round_idx == 0:
                # First round: just the problem with tools
                problem_message = MathPromptTemplates.PROBLEM_FORMAT.format(
                    problem=problem
                )
                response = self.reasoner.run(problem_message)
                solution = response.content
            else:
                # Subsequent rounds: include previous solution with critique
                critique_prompt = get_critic_prompt(
                    problem=problem,
                    previous_solution=state.answer,
                    round_idx=round_idx,
                )
                response = self.reasoner.run(critique_prompt)
                solution = response.content

            # Extract and update solution
            state.answer = MathTools.extract_final_answer(solution)
            state.answer_list.append(state.answer)
            self.session_state[problem] = state

            yield RunResponse(
                run_id=self.run_id,
                content=f"Round {round_idx + 1} solution: {self.session_state[problem].answer}",
                event=RunEvent.workflow_started
            )

        # Select final solution
        if self.selection_strategy == "last":
            selected_solution = self.session_state[problem].answer_list[-1]
        elif self.selection_strategy == "agent" and len(self.session_state[problem].answer_list) > 1:
            selected_solution = self._select_solution(problem, self.session_state[problem].answer_list)
        else:
            selected_solution = "\n".join(self.session_state[problem].answer_list)

        # Verify and store solution
        state.answer = self._verify_solution(problem, selected_solution)
        self.session_state[problem] = state

        yield RunResponse(
            run_id=self.run_id,
            content=self.session_state[problem].answer,
            event=RunEvent.workflow_completed
        )

    def _select_solution(self, problem: str, solutions: List[str]) -> str:
        """
        Select the best solution from multiple candidates.
        """
        print("=====Selecting the best solution...=====")

        # Format solutions for the selector prompt
        formatted_solutions = "\n\n".join(
            [f"Solution {i + 1}:\n{solution}" for i, solution in enumerate(solutions)]
        )
        
        selection_prompt = MathPromptTemplates.SELECTION_PROMPT.format(
            problem=problem, solutions=formatted_solutions
        )

        response = self.reasoner.run(selection_prompt)
        result = response.content

        # Parse the result to find the selected solution
        match = re.search(r"Solution\s+(\d+)\s+is the best", result, re.IGNORECASE)
        
        if match:
            solution_idx = int(match.group(1)) - 1
            if 0 <= solution_idx < len(solutions):
                return solutions[solution_idx]

        # Default to the last solution if parsing fails
        return solutions[-1]

    def _verify_solution(
        self, problem: str, solution: str, max_attempts: int = 3
    ) -> str:
        """
        Verify the solution and attempt to correct it if needed.

        Args:
            problem: The original problem
            solution: The proposed solution
            max_attempts: Maximum verification attempts

        Returns:
            The verified (and potentially corrected) solution
        """
        print("=====Verifying the solution...=====")
        current_solution = solution

        for attempt in range(max_attempts):
            # Run verification check
            verification_prompt = MathPromptTemplates.VERIFICATION_PROMPT.format(
                problem=problem, solution=current_solution
            )
            response = self.reasoner.run(verification_prompt)
            verification_result = response.content

            # Check if verification was successful
            if "VERIFICATION: CORRECT" in verification_result:
                print(f"Solution verified as correct after {attempt + 1} attempts.")
                return current_solution

            # Extract revised solution if available
            revised_match = re.search(
                r"REVISED SOLUTION:(.*?)(?=$|VERIFICATION:)", verification_result, re.DOTALL
            )

            if revised_match:
                current_solution = revised_match.group(1).strip()
                print(f"Solution revised in attempt {attempt + 1}.")
            else:
                return f"{current_solution}\n\n===== VERIFICATION NOTES =====\n{verification_result}"

        # Return the best solution after max attempts
        return current_solution
