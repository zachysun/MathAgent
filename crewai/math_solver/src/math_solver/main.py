from random import randint
from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from math_solver.crews.math_crew.math_crew import MathCrew
from math_solver.utils import extract_answer


class MathSolverState(BaseModel):
    rounds: int = 3
    problem: str = """
        Let $ABC$ be a triangle with $BC=108$, $CA=126$, and $AB=39$. 
        Point $X$ lies on segment $AC$ such that $BX$ bisects $\angle CBA$. 
        Let $\omega$ be the circumcircle of triangle $ABX$. 
        Let $Y$ be a point on $\omega$ different from $X$ such that $CX=CY$. 
        Line $XY$ meets $BC$ at $E$. 
        The length of the segment $BE$ can be written as $\frac{m}{n}$, where $m$ and $n$ are coprime positive integers. 
        Find $m+n$.
        """
    answer_list: list = []
    answer: int = 0


class MathSovlerFlow(Flow[MathSolverState]):

    @start()
    def multi_round_reasoning(self):
        for _ in range(self.state.rounds):
            result = MathCrew().base_reasoner_crew().kickoff(inputs={
                "problem": self.state.problem,
            })
            self.state.answer = extract_answer(result.raw)
            self.state.answer_list.append(self.state.answer)
            
    @listen(multi_round_reasoning)
    def selection(self):
        result = MathCrew().selector_crew().kickoff(inputs={
                "problem": self.state.problem,
                "answer_list": self.state.answer_list,
            })
        self.state.answer = extract_answer(result.raw)

    @listen(selection)
    def verification(self):
        result = MathCrew().validator_crew().kickoff(inputs={
                "problem": self.state.problem,
                "answer": self.state.answer,
            })
        self.state.answer = extract_answer(result.raw)

def kickoff():
    math_solver_flow = MathSovlerFlow()
    math_solver_flow.kickoff()


def plot():
    math_solver_flow = MathSovlerFlow()
    math_solver_flow.plot()


if __name__ == "__main__":
    kickoff()
