import os
import re
from typing import Optional, List
from dotenv import load_dotenv

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.toolkits import MathToolkit

from utils import MathTools
from prompts import (
    get_system_prompt,
    format_problem,
    get_critic_prompt,
    get_selector_prompt,
    get_verification_prompt,
)


class MathReasonerAgent:
    """
    A math reasoning agent based on CAMEL.
    """

    def __init__(
        self,
        model_platform: ModelPlatformType = ModelPlatformType.OPENAI,
        model_type: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        use_vllm_local: bool = False,
        vllm_model_path: Optional[str] = None,
        temperature: float = 1.0,
        token_limit: int = 8192,
        selection_strategy: str = "agent",
    ):
        """
        Initialize the MathReasonerAgent.

        Args:
            model_platform: The platform of the model (OPENAI, OPENAI_COMPATIBLE_MODEL, VLLM, etc.)
            model_type: The specific model to use
            api_key: API key for online models (will override .env if provided)
            api_base_url: Base URL for API (will override .env if provided)
            use_vllm_local: Whether to use local vLLM model (default: False)
            vllm_model_path: Path to local vLLM model (default: None)
            temperature: Temperature for generation (default: 1.0)
            token_limit: Maximum token limit (default: 8192)
            selection_strategy: Strategy for selecting the best solution ("agent", "last", "all")
        """
        # Load environment variables
        load_dotenv(override=True)

        # Initialize API key and base URL variables
        api_key_for_model = None
        api_base_url_for_model = None

        # If not using local vLLM model, handle API key and base URL
        if not use_vllm_local:
            if model_platform == ModelPlatformType.OPENAI:
                api_key_for_model = (api_key if api_key else os.environ.get("OPENAI_API_KEY"))
                api_base_url_for_model = (api_base_url if api_base_url else os.environ.get("OPENAI_API_BASE_URL"))
                
            elif model_platform == ModelPlatformType.OPENAI_COMPATIBLE_MODEL:
                api_key_for_model = (api_key if api_key else os.environ.get("OPENAI_COMPATIBLE_MODEL_API_KEY"))
                api_base_url_for_model = (api_base_url if api_base_url else os.environ.get("OPENAI_COMPATIBLE_MODEL_API_BASE_URL"))
            
            elif model_platform == ModelPlatformType.DEEPSEEK:
                api_key_for_model = (api_key if api_key else os.environ.get("DEEPSEEK_API_KEY"))
                api_base_url_for_model = (api_base_url if api_base_url else os.environ.get("DEEPSEEK_API_BASE_URL"))
            
            elif model_platform == ModelPlatformType.SILICONFLOW:
                api_key_for_model = (api_key if api_key else os.environ.get("SILICONFLOW_API_KEY"))
                api_base_url_for_model = (api_base_url if api_base_url else os.environ.get("SILICONFLOW_API_BASE_URL"))

        # Configure model
        self.model_config_dict = {
            "temperature": temperature,
            "max_tokens": token_limit,
        }

        self.model_platform = model_platform
        self.model_type = model_type
        self.api_key = api_key_for_model
        self.api_base_url = api_base_url_for_model
        self.use_vllm_local = use_vllm_local
        self.vllm_model_path = vllm_model_path
        self.token_limit = token_limit
        self.selection_strategy = selection_strategy

        # Initialize LLM
        self._initialize_llm()

        # Initialize math tools
        self.math_tools = MathToolkit().get_tools()

        # Create system message for deductive reasoning using prompts library
        self.system_message = get_system_prompt()

        # Initialize chat agent
        self.agent = ChatAgent(
            system_message=self.system_message,
            model=self.model,
            tools=self.math_tools,
            token_limit=token_limit,
        )

    def _initialize_llm(self, temperature=None):
        """
        Initialize the LLM.
        """
        model_config = self.model_config_dict.copy()
        if temperature is not None:
            model_config["temperature"] = temperature

        if self.use_vllm_local and self.vllm_model_path:
            # Local vLLM model
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.VLLM,
                model_type=self.vllm_model_path,
                model_config_dict=model_config,
            )
        else:
            # Online model
            if not self.model_type:
                if self.model_platform == ModelPlatformType.OPENAI:
                    self.model_type = "gpt-4o-mini"
                elif self.model_platform == ModelPlatformType.OPENAI_COMPATIBLE_MODEL:
                    self.model_type = "gpt-4o-mini"

            self.model = ModelFactory.create(
                model_platform=self.model_platform,
                model_type=self.model_type,
                api_key=self.api_key,
                url=self.api_base_url,
                model_config_dict=model_config,
            )

    def _create_new_agent(self, system_prompt, temperature=None):
        """
        Create a new agent with a different system prompt.

        Args:
            system_prompt: The system prompt for the new agent
            temperature: Optional temperature for the new agent

        Returns:
            A new ChatAgent
        """
        # Initialize a new model with different temperature
        pre_temperature = self.model_config_dict["temperature"]
        self._initialize_llm(
            temperature=temperature if temperature is not None else pre_temperature
        )

        # Create a new agent with the new system prompt
        return ChatAgent(
            system_message=system_prompt, model=self.model, token_limit=self.token_limit
        )

    def solve_problem(self, problem: str, num_rounds: int = 3) -> str:
        """
        Solve a mathematical problem using multi-round reasoning.

        Args:
            problem: The mathematical problem to solve.
            num_rounds: Number of reasoning rounds (default: 3)

        Returns:
            The solution to the problem.
        """
        all_solutions = []
        prev_solution = None

        print(f"Solving problem with {num_rounds} rounds of reasoning...")

        # Multi-round reasoning
        for round_idx in range(num_rounds):
            print(f"\nRound {round_idx + 1}/{num_rounds}")

            # Format the problem message using the prompts library
            if round_idx == 0:
                # First round: just the problem
                problem_message = format_problem(problem)
                agent = self.agent
            else:
                # Subsequent rounds: include previous solution with critique
                critic_prompt = get_critic_prompt(problem, prev_solution, round_idx)
                agent = self._create_new_agent(critic_prompt, temperature=0.9)
                problem_message = f"Provide a different solution to this problem."

            # Get response from the agent
            response = agent.step(problem_message)
            solution = response.msg.content

            final_answer = MathTools.extract_final_answer(solution)
            all_solutions.append(final_answer)
            prev_solution = final_answer

            # Only keep memory of the previous round's final answer
            prev_solution = final_answer if final_answer else solution

            print(f"Extracted answer: {final_answer}")

        # Choose final solution based on selection strategy
        if self.selection_strategy == "last":
            final_solution = all_solutions[-1]
        elif self.selection_strategy == "all":
            final_solution = all_solutions
        else:  # Use selection strategy "agent" (default)
            # Select the best solution
            final_solution = self._select_solution(problem, all_solutions)

        # Verify the solution
        if self.selection_strategy != "all":
            verified_solution = self._verify_solution(problem, final_solution)
        else:
            verified_solution = final_solution

        return verified_solution

    def _select_solution(self, problem: str, solutions: List[str]) -> str:
        """
        Select the best solution from multiple candidates.

        Args:
            problem: The original problem
            solutions: List of solutions to choose from

        Returns:
            The best solution
        """
        selector_prompt = get_selector_prompt()
        selector_agent = self._create_new_agent(selector_prompt, temperature=0.3)

        # Format the solutions into a single message
        solutions_text = "\n\n".join(
            [f"Solution {i + 1}:\n{solution}" for i, solution in enumerate(solutions)]
        )
        selection_message = f"Problem: {problem}\n\n{solutions_text}\n\nPlease select the best solution and explain why."

        # Get response from the selector agent
        response = selector_agent.step(selection_message)
        selection_result = response.msg.content

        # Extract the selected solution
        for i, solution in enumerate(solutions):
            if f"Solution {i + 1}" in selection_result and "best" in selection_result:
                return solution

        # Default to the last solution if no clear selection
        return solutions[-1]

    def _verify_solution(
        self, problem: str, solution: str, max_attempts: int = 3
    ) -> str:
        """
        Verify the solution and refine if necessary.

        Args:
            problem: The original problem
            solution: The solution to verify
            max_attempts: Maximum number of verification attempts

        Returns:
            The verified solution
        """
        print("=====Verifying the solution...=====")
        verification_prompt = get_verification_prompt()
        verification_agent = self._create_new_agent(verification_prompt, temperature=0.3)

        current_solution = solution

        for attempt in range(max_attempts):
            verification_message = (
                f"Problem: {problem}\n\nSolution to verify:\n{current_solution}"
            )

            # Get response from the verification agent
            response = verification_agent.step(verification_message)
            verification_result = response.msg.content

            # Check if the solution is verified
            if "VERIFICATION: CORRECT" in verification_result:
                print(f"Solution verified as correct after {attempt + 1} attempts.")
                return current_solution

            # If not verified, extract the revised solution
            revised_match = re.search(
                r"REVISED SOLUTION:(.*?)(?=$|VERIFICATION:)", verification_result, re.DOTALL
            )

            if revised_match:
                current_solution = revised_match.group(1).strip()
                print(f"Solution revised in attempt {attempt + 1}.")
            else:
                return f"{current_solution}\n\n===== VERIFICATION NOTES =====\n{verification_result}"

        # If max attempts reached without verification, return the last solution
        return current_solution
