import os
import re
import argparse
from typing import Optional, List
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLM

from langchain.agents import AgentExecutor, create_openai_functions_agent

from utils import MathTools, get_math_tools
from prompts import MathPromptTemplates, get_critic_prompt


class MathReasonerAgent:
    """
    A math reasoning agent based on Langchain.
    """

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        use_local_model: bool = False,
        local_model_path: Optional[str] = None,
        temperature: float = 1.0,
        token_limit: int = 8192,
        selection_strategy: str = "agent",
    ):
        """
        Initialize the MathReasonerAgent.

        Args:
            model_name: The name of the model to use (default: "deepseek-chat")
            api_key: API key for model provider (will override .env if provided)
            api_base_url: Base URL for API (will override .env if provided)
            use_local_model: Whether to use local model (default: False)
            local_model_path: Path to local model (default: None)
            temperature: Temperature for generation (default: 1.0)
            token_limit: Maximum token limit (default: 8192)
            selection_strategy: Strategy for selecting the best solution ("agent", "last", "all")
        """
        # Load environment variables
        load_dotenv(override=True)

        # Initialize API key and base URL
        self.api_key = (api_key if api_key else os.environ.get("DEEPSEEK_API_KEY"))
        self.api_base_url = (api_base_url if api_base_url else os.environ.get("DEEPSEEK_API_BASE_URL"))

        self.model_name = model_name
        self.use_local_model = use_local_model
        self.local_model_path = local_model_path
        self.temperature = temperature
        self.token_limit = token_limit
        self.selection_strategy = selection_strategy

        # Get math tools
        self.math_tools = get_math_tools()
        # Create system message
        self.system_message = SystemMessage(content=MathPromptTemplates.SYSTEM_PROMPT)

        # Initialize LLM
        self._initialize_llm()
        # Initialize the agent
        self._initialize_agent()
        

    def _initialize_llm(self) -> None:
        """Initialize the llm."""
        if self.use_local_model and self.local_model_path:
            # Initialize local model using VLLM
            self.llm = VLLM(
                model=self.local_model_path,
                temperature=self.temperature,
                max_tokens=self.token_limit,
            )
        else:
            # Initialize ChatOpenAI model
            self.llm = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.api_base_url,
                temperature=self.temperature,
                max_tokens=self.token_limit,
            )

    def _initialize_agent(self) -> None:
        """Initialize the agent."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_message.content),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self.agent = create_openai_functions_agent(
            llm=self.llm, 
            tools=self.math_tools, 
            prompt=prompt
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.math_tools,
            verbose=True,
            handle_parsing_errors=True,
        )

        init_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_message.content),
                ("human", "{input}"),
            ]
        )
        self.base_chain = init_prompt | self.llm | StrOutputParser()

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

            if round_idx == 0:
                # First round: just the problem
                problem_message = MathPromptTemplates.PROBLEM_FORMAT.format(
                    problem=problem
                )
                response = self.agent_executor.invoke({"input": problem_message})
                solution = response["output"]
            else:
                # Subsequent rounds: include previous solution with critique
                critic_prompt = get_critic_prompt(
                    problem=problem,
                    previous_solution=prev_solution,
                    round_idx=round_idx,
                )

                formatted_input = critic_prompt.format(
                    problem=problem, 
                    previous_solution=prev_solution
                )
                solution = self.base_chain.invoke({"input": formatted_input})

            # Extract final answer
            final_answer = MathTools.extract_final_answer(solution)
            all_solutions.append(final_answer)
            prev_solution = final_answer

            print(f"Extracted answer: {final_answer}")

        # Select the best solution
        if self.selection_strategy == "last":
            selected_solution = all_solutions[-1]
        elif self.selection_strategy == "agent" and len(all_solutions) > 1:
            selected_solution = self._select_solution(problem, all_solutions)
        else:
            selected_solution = (
                "\n\n===== ALL SOLUTIONS =====\n\n"
                + "\n\n-----\n\n".join(all_solutions)
            )

        # Verify the solution
        final_solution = self._verify_solution(problem, selected_solution)
        return final_solution

    def _select_solution(self, problem: str, solutions: List[str]) -> str:
        """
        Select the best solution from multiple candidates.

        Args:
            problem: The original problem
            solutions: List of solution candidates

        Returns:
            The selected solution
        """
        print("=====Selecting the best solution...=====")

        # Format solutions for the selector prompt
        formatted_solutions = "\n\n".join(
            [f"Solution {i + 1}:\n{solution}" for i, solution in enumerate(solutions)]
        )

        selection_prompt = ChatPromptTemplate.from_template(
            MathPromptTemplates.SELECTION_PROMPT
        )

        selection_chain = selection_prompt | self.llm | StrOutputParser()

        evaluation = selection_chain.invoke(
            {"problem": problem, "solutions": formatted_solutions}
        )

        # Parse the evaluation to find the selected solution
        match = re.search(r"Solution\s+(\d+)\s+is the best", evaluation, re.IGNORECASE)

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

        verification_prompt = ChatPromptTemplate.from_template(
            MathPromptTemplates.VERIFICATION_PROMPT
        )
        verification_chain = verification_prompt | self.llm | StrOutputParser()

        for attempt in range(max_attempts):
            verification = verification_chain.invoke(
                {"problem": problem, "solution": current_solution}
            )

            # Check if the verification was successful
            if "VERIFICATION: CORRECT" in verification:
                print(f"Solution verified as correct after {attempt + 1} attempts.")
                return current_solution

            # Extract the revised solution if available
            revised_match = re.search(
                r"REVISED SOLUTION:(.*?)(?=$|VERIFICATION:)", verification, re.DOTALL
            )

            if revised_match:
                current_solution = revised_match.group(1).strip()
                print(f"Solution revised in attempt {attempt + 1}.")
            else:
                return f"{current_solution}\n\n===== VERIFICATION NOTES =====\n{verification}"

        # Return the best solution after max attempts
        return current_solution
