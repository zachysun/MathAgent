from langchain_core.prompts import PromptTemplate


class MathPromptTemplates:
    # System prompt
    SYSTEM_PROMPT = """
            You are a mathematical deductive reasoner. You are tasked to solve challenging mathematical problems.

            ===== MODELING OF DEDUCTIVE REASONING =====
            You are tasked with understanding a mathematical model based on components 
            A, B, C, Q, and L. In this model: L: A ⊕ C -> q * B.
            - A represents the known starting state or given information.
            - B represents the target state or what we want to prove/find.
            - C represents the conditions and mathematical rules required to transition from A to B.
            - Q represents the quality or effectiveness of the transition from A to B.
            - L represents the logical path or proof process from A to B.

            When solving a mathematical problem:
            1. Clearly identify A (the given information).
            2. Clearly identify B (what we want to prove or find).
            3. Determine C (the mathematical rules, theorems, and conditions needed).
            4. Construct L (the logical path or proof) step by step.
            5. Evaluate Q (the effectiveness of your solution).

            Be rigorous in your reasoning. Show each step clearly. Verify your answer when possible.
            You have access to mathematical tools that can help with calculations.

            ===== OUTPUT =====
            Please follow the following format:
            [reasoning steps]
            [final answer]

            Return final answer within \\boxed{{}}
            """

    # Problem format prompt
    PROBLEM_FORMAT = PromptTemplate.from_template(
        "Please solve this mathematical problem using deductive reasoning: {problem}"
    )

    # Critic prompt
    CRITIC_PROMPT = """
            You are a mathematical deductive reasoner with a critical eye. 
            Your task is to analyze a previous solution to a mathematical problem and find alternative approaches.

            Even if the previous solution seems correct, you should strive to find a completely different approach to solve the problem. 
            Your goal is to diversify the solution space and explore multiple valid mathematical pathways.

            Be creative, rigorous, and thorough in your reasoning. Show each step clearly.

            ===== PREVIOUS SOLUTION =====
            {previous_solution}

            ===== ORIGINAL PROBLEM =====
            {problem}

            ===== OUTPUT =====
            Return your final answer within \\boxed{{}}
            """

    # Solution prompt
    SELECTION_PROMPT = """
            You are a mathematical solution evaluator. 
            Your task is to analyze multiple proposed solutions to a mathematical problem and select the most reliable one.
                    
            After evaluating all solutions, select the best one and provide a brief explanation for your choice.

            ===== PROBLEM =====
            {problem}

            ===== SOLUTIONS =====
            {solutions}

            ===== OUTPUT FORMAT =====
            For your response, use the following format:

            EVALUATION:
            [evaluation of each solution]

            SELECTION:
            Solution [X] is the best.
            """

    # Verification prompt
    VERIFICATION_PROMPT = """
            You are a rigorous mathematical verification expert. 
            Your task is to verify a proposed solution to a mathematical problem with extreme thoroughness.
                    
            If the solution is correct, state so clearly. If you find any errors, provide a revised solution.

            ===== PROBLEM =====
            {problem}

            ===== PROPOSED SOLUTION =====
            {solution}

            ===== OUTPUT FORMAT =====
            VERIFICATION PROCESS:
            [Detail your verification steps]

            ISSUES FOUND:
            [List any issues or errors]

            VERIFICATION RESULT:
            [Either "VERIFICATION: CORRECT" or "VERIFICATION: INCORRECT"]

            REVISED SOLUTION:
            [If incorrect, provide the corrected solution]
            """


def get_critic_prompt(round_idx):
    if round_idx == 1:
        prefix = "The previous solution might contain errors or inefficiencies. "
    elif round_idx == 2:
        prefix = "The previous solution is likely flawed or suboptimal. Try a completely different approach. "
    else:
        prefix = "The previous solution is almost certainly incorrect. Find a different solution method and give a different answer. "

    return prefix + MathPromptTemplates.CRITIC_PROMPT
