from dotenv import load_dotenv

from math_agent import MathReasoningWorkflow


PROBLEM_EXAMPLE = """
Let $ABC$ be a triangle with $BC=108$, $CA=126$, and $AB=39$. 
Point $X$ lies on segment $AC$ such that $BX$ bisects $\angle CBA$. 
Let $\omega$ be the circumcircle of triangle $ABX$. 
Let $Y$ be a point on $\omega$ different from $X$ such that $CX=CY$. 
Line $XY$ meets $BC$ at $E$. 
The length of the segment $BE$ can be written as $\frac{m}{n}$, where $m$ and $n$ are coprime positive integers. 
Find $m+n$.
"""


def main():
    """Run the math agent."""
    # load environment variables
    load_dotenv(override=True)

    # Create the workflow
    workflow = MathReasoningWorkflow(
        model_name="deepseek-chat",
        temperature=1.0,
        token_limit=8192
    )

    # Run the workflow
    responses = workflow.run(problem=PROBLEM_EXAMPLE, num_rounds=3)
    for response in responses:
        print(f"\n{response.content}")


if __name__ == "__main__":
    print("=" * 70)
    print("Math reasoning workflow usage examples")
    print("=" * 70)

    main()
