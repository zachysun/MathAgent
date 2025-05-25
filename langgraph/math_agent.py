from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from nodes import MultiRoundNode, SelectionNode, VerificationNode

class AgentState(TypedDict):
    problem: str
    num_rounds: int
    answer: Optional[str]
    answer_list: List[str]
    verification_status: Optional[str]
    verification_notes: Optional[str]

def create_math_agent():
    """Create math agent workflow"""
    # Initialize nodes
    multi_round = MultiRoundNode()
    selection = SelectionNode()
    verification = VerificationNode()

    # Build graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("multi_round", multi_round.run)
    workflow.add_node("selection", selection.run)
    workflow.add_node("verification", verification.run)

    # Define edges
    workflow.add_edge("multi_round", "selection")
    workflow.add_edge("selection", "verification")
    workflow.add_edge("verification", END)

    # Set entry point
    workflow.set_entry_point("multi_round")
    
    # Compile graph
    return workflow.compile()

def solve_problem(problem: str, num_rounds: int = 3):
    """Run math agent workflow"""
    if not problem:
        raise ValueError("Problem cannot be empty")

    # Initialize state
    state = AgentState(
        problem=problem.strip(),
        num_rounds=num_rounds,
        answer=None,
        answer_list=[],
        verification_status=None,
        verification_notes=None
    )

    # Create and run workflow
    app = create_math_agent()
    result = app.invoke(state)

    return result["answer"]
