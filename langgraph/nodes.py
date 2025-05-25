import re
from typing import Dict, Any
from utils import call_llm, extract_final_answer

class MultiRoundNode:
    """Multi-round reasoning node"""
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-round reasoning"""
        
        print('=' * 10 + 'Multi-round reasoning' + '=' * 10)
        
        problem = state["problem"]
        num_rounds = state["num_rounds"]
        answer_list = []
        prev_answer = None
        
        for round_idx in range(num_rounds):
            if round_idx == 0:
                prompt = f"""
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
                
                The problem is: 
                
                {problem}
                
                Please return the answer within \\boxed{{}}."""
            else:
                prompt = f"""Re-examine the following math problem and analyze the previous solution:
                
                Problem: {problem}
                Previous solution: {prev_answer}
                
                Please provide an improved solution and return within \\boxed{{}}"""
                
            response = call_llm(prompt)            
            final_answer = extract_final_answer(response)
            answer_list.append(final_answer)
            prev_answer = final_answer
            
        return {
            "answer_list": answer_list,
            "problem": problem,
            "num_rounds": num_rounds
        }

class SelectionNode:
    """Solution selection node"""
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best solution from multiple rounds"""
        
        print('=' * 10 + 'Selecting' + '=' * 10)
        print('Current state: \n', state)
        
        problem = state["problem"]
        answer_list = state["answer_list"]
        
        if len(answer_list) <= 1:
            return {"answer": answer_list[0] if answer_list else None}
            
        formatted_answers = "\n\n".join(
            [f"Solution {i + 1}:\n{sol}" for i, sol in enumerate(answer_list)]
        )
        
        prompt = f"""Select the best solution from the following options:
        
        Problem: {problem}
        
        {formatted_answers}
        
        Compare solutions based on accuracy and completeness.
        Respond in format: "Best solution is: Solution [number]" """
        
        response = call_llm(prompt)
        match = re.search(r"Solution\s*(\d+)", response)
        
        if match:
            selected_idx = int(match.group(1)) - 1
            if 0 <= selected_idx < len(answer_list):
                return {"answer": answer_list[selected_idx]}
        
        return {"answer": answer_list[-1]}

class VerificationNode:
    """Solution verification node"""
    
    def __init__(self, max_attempts=1):
        self.max_attempts = max_attempts
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Verify the selected solution"""
        
        print('=' * 10 + 'Verifying' + '=' * 10)
        
        problem = state["problem"]
        answer = state["answer"]
        
        for _ in range(self.max_attempts):
            prompt = f"""Verify if the following solution to the math problem is correct:
            
            Problem: {problem}
            Solution: {answer}
            
            Check each step of the solution. If errors are found, provide a corrected solution.
            
            Response format:
            - If correct: "Verification: CORRECT"
            - If incorrect: "Verification: INCORRECT\nCorrected solution: [corrected solution]" """
            
            response = call_llm(prompt)
            
            if "Verification: CORRECT" in response:
                return {
                    "answer": answer,
                    "verification_status": "verified",
                    "verification_notes": "Answer verified as correct"
                }
            
            revised_match = re.search(r"Corrected solution[:：]\s*(.+)", response, re.DOTALL)
            if revised_match:
                answer = revised_match.group(1).strip()
                continue
                
        return {
            "answer": answer,
            "verification_status": "unverified",
            "verification_notes": f"Solution not fully verified after {self.max_attempts} attempts"
        }