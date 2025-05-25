from pocketflow import Flow
from nodes import MultiRound, Selection, Verification

def create_math_flow():
    """Create and configure the problem solving workflow
    
    Args:
        num_rounds (int): Number of reasoning rounds (default: 3)
        
    Returns:
        Flow: Configured math problem solving flow
    """
    # Create nodes
    multi_round = MultiRound()
    selection = Selection()
    verification = Verification()
    
    # Connect nodes
    multi_round >> selection >> verification
    
    # Create flow starting with multi_round node
    math_flow = Flow(start=multi_round)
    
    return math_flow

def solve_problem(problem, num_rounds=3):
    """Run the problem solving workflow
    
    Args:
        problem (str): The math problem to solve
        num_rounds (int): Number of reasoning rounds (default: 3)
        
    Returns:
        str: The final solution to the problem
        
    Raises:
        ValueError: If problem is empty or invalid
    """
    if not problem:
        raise ValueError("Problem cannot be empty")
        
    problem = str(problem).strip()
    answer = None
    answer_list = []
    
    # Initialize shared data
    shared = {
        "problem": problem,
        "num_rounds": num_rounds,
        "answer": answer,
        "answer_list": answer_list
    }
    
    # Create and run flow
    flow = create_math_flow()
    flow.run(shared)
    
    solution = shared["answer"]
    
    return solution

