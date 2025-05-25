import os
import asyncio
from typing import Optional
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console

from prompts import MathPromptTemplates


class MathReasonerAgent:
    """
    A math reasoning agent based on AutoGen.
    """

    def __init__(
        self,
        model_name: str = "DeepSeek-V3",
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        temperature: float = 1.0,
        token_limit: int = 8192,
    ):
        """
        Initialize the MathReasonerAgent.

        Args:
            model_name: The name of the model to use (default: "DeepSeek-V3")
            api_key: API key for model provider (will override .env if provided)
            api_base_url: Base URL for API (will override .env if provided)
            temperature: Temperature for generation (default: 1.0)
            token_limit: Maximum token limit (default: 8192)
        """
        # Load environment variables
        load_dotenv(override=True)

        # Initialize API key and base URL variables
        self.api_key = (api_key if api_key else os.environ.get("OPENAI_COMPATIBLE_MODEL_API_KEY"))
        self.api_base_url = (api_base_url if api_base_url else os.environ.get("OPENAI_COMPATIBLE_MODEL_API_BASE_URL"))

        self.model_name = model_name
        self.temperature = temperature
        self.token_limit = token_limit

        # Initialize the agent
        self._initialize_agent()

    def _initialize_agent(self) -> None:
        """Initialize the agents"""
        # Initialize the LLM
        model_client = OpenAIChatCompletionClient(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.api_base_url,
            temperature=self.temperature,
            max_tokens=self.token_limit,
            model_info={
                "json_output": False,
                "function_calling": False,
                "vision": False,
                "family": self.model_name,
            },
        )

        # Initialize the agents
        self.base_agent = AssistantAgent(
            name="BaseMathReasoner",
            model_client=model_client,
            system_message=MathPromptTemplates.SYSTEM_PROMPT,
        )

        self.critic_agent = AssistantAgent(
            name="MathReasonerCritic",
            model_client=model_client,
            system_message=MathPromptTemplates.CRITIC_PROMPT,
        )

        self.verification_agent = AssistantAgent(
            name="MathReasonerVerification",
            model_client=model_client,
            system_message=MathPromptTemplates.VERIFICATION_PROMPT,
        )

    async def solve_problem(self, problem: str) -> str:
        max_msg_termination = MaxMessageTermination(max_messages=8)
        math_reasoner_team = RoundRobinGroupChat(
            [
                self.base_agent,
                self.critic_agent,
                self.verification_agent,
            ],
            termination_condition=max_msg_termination,
        )
        await Console(math_reasoner_team.run_stream(task=problem))


def main():
    PROBLEM_EXAMPLE = """
    Let $ABC$ be a triangle with $BC=108$, $CA=126$, and $AB=39$. 
    Point $X$ lies on segment $AC$ such that $BX$ bisects $\angle CBA$. 
    Let $\omega$ be the circumcircle of triangle $ABX$. 
    Let $Y$ be a point on $\omega$ different from $X$ such that $CX=CY$. 
    Line $XY$ meets $BC$ at $E$. 
    The length of the segment $BE$ can be written as $\frac{m}{n}$, where $m$ and $n$ are coprime positive integers. 
    Find $m+n$.
    """
    math_agent = MathReasonerAgent()
    asyncio.run(math_agent.solve_problem(PROBLEM_EXAMPLE))


if __name__ == "__main__":
    main()
