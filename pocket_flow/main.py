import argparse
from flow import solve_problem

def main():
    parser = argparse.ArgumentParser(description="Math Problem Solver using PocketFlow")
    parser.add_argument("--problem", type=str, help="The math problem to solve")
    parser.add_argument(
        "--rounds", 
        type=int, 
        default=3, 
        help="Number of reasoning rounds (default: 3)"
    )
    args = parser.parse_args()

    if args.problem:
        solution = solve_problem(args.problem, num_rounds=args.rounds)
        print(f"\nFinal Solution:\n{solution}")
    else:
        print("Problem cannot be empty")

if __name__ == "__main__":
    problem = """
Find all real solutions to the equation:  
\[ \sqrt{x + 3} + \sqrt{x - 2} = 5 \].
"""

    print("=" * 70)
    print("MathReasonerAgent usage examples")
    print("=" * 70)
    
    print(f"\nExample: {problem}")
    answer = solve_problem(problem)
    print(f"Answer: {answer}")
