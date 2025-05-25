import re
from pocketflow import Node
from utils import call_llm, extract_final_answer

class MultiRound(Node):
    """Multi-round mathe reasoning node"""
    
    def __init__(self):
        super().__init__()
        
    def prep(self, shared):
        """Prepare input data"""
        
        return {
            "problem": shared["problem"],
            "num_rounds": shared["num_rounds"]
        }
        
    def exec(self, prep_res):
        """Execute multi-round reasoning"""
        
        problem = prep_res["problem"]
        num_rounds = prep_res["num_rounds"]
        answer_list = []
        prev_answer = None
        
        for round_idx in range(num_rounds):
            if round_idx == 0:
                # First round reasoning
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
                # Subsequent rounds include critique of previous solution
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
        }

    def post(self, shared, prep_res, exec_res):
        """Store reasoning results"""
        
        shared["answer_list"] = exec_res["answer_list"]
        
        print("\n===== Multi-round Reasoning Results =====\n")
        for i, solution in enumerate(exec_res["answer_list"]):
            print(f"Round {i+1}: {solution}")
        print("\n=======================================\n")
        
        return "default"
        


class Selection(Node):
    """Solution selection node"""
    
    def __init__(self):
        super().__init__()
    
    def prep(self, shared):
        """Prepare input data"""
        
        return {
            "problem": shared["problem"],
            "answer_list": shared["answer_list"]
        }
    
    def exec(self, prep_res):
        """Selecting the best solution"""
        
        problem = prep_res["problem"]
        answers_list = prep_res["answer_list"]
        
        if len(answers_list) <= 1:
            return answers_list[0] if answers_list else None
            
        # Build selection prompt
        formatted_answers = "\n\n".join(
            [f"Solution {i + 1}:\n{sol}" for i, sol in enumerate(answers_list)]
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
            if 0 <= selected_idx < len(answers_list):
                return answers_list[selected_idx]
        
        # Default to last solution
        return answers_list[-1]

    def post(self, shared, prep_res, exec_res):
        """Store selection result"""
        shared["answer"] = exec_res
        
        print("\n===== Selected Best Solution =====\n")
        print(exec_res)
        print("\n=================================\n")
        
        return "default"


class Verification(Node):
    """Solution verification"""
    
    def __init__(self, max_attempts=1):
        super().__init__()
        self.max_attempts = max_attempts

    def prep(self, shared):
        """Prepare input data"""
        
        return {
            "problem": shared["problem"],
            "answer": shared["answer"]
        }
        
    def exec(self, prep_res):
        """Verify solution"""
        problem = prep_res["problem"]
        answer = prep_res["answer"]
        
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
                    "status": "verified",
                    "answer": answer,
                    "notes": "Answer verified as correct"
                }
            
            # Try to extract corrected solution
            revised_match = re.search(r"Corrected solution[:：]\s*(.+)", response, re.DOTALL)
            if revised_match:
                answer = revised_match.group(1).strip()
                continue
                
        return {
            "status": "unverified",
            "answer": answer,
            "notes": f"Solution not fully verified after {self.max_attempts} attempts"
        }
        
    def post(self, shared, prep_res, exec_res):
        """Store verification results"""
        
        shared["answer"] = exec_res["answer"]
        shared["verification_status"] = exec_res["status"]
        shared["verification_notes"] = exec_res["notes"]
        
        print("\n===== Verification Results =====\n")
        print(f"Status: {shared['verification_status']}")
        print(f"Answer: {shared['answer']}")
        print(f"Notes: {shared['verification_notes']}")
        print("\n===============================\n")
        
        return "default"
