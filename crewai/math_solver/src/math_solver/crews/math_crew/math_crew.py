from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

@CrewBase
class MathCrew():
    """MathCrew crew"""
    agents: List[BaseAgent]
    tasks: List[Task]
    
    agents_config = 'config/agents.yaml' 
    tasks_config = 'config/tasks.yaml' 
    
    @agent
    def base_reasoner(self) -> Agent:
        return Agent(
            config=self.agents_config['base_reasoner'],
            verbose=True
        )

    @agent
    def selector(self) -> Agent:
        return Agent(
            config=self.agents_config['selector'],
            verbose=True
        )
        
    @agent
    def validator(self) -> Agent:
        return Agent(
            config=self.agents_config['validator'],
            verbose=True
        )

    @task
    def base_solving_task(self) -> Task:
        return Task(
            config=self.tasks_config['base_solving_task']
        )

    @task
    def selection_task(self) -> Task:
        return Task(
            config=self.tasks_config['selection_task'],
        )
    
    @task
    def validation_task(self) -> Task:
        return Task(
            config=self.tasks_config['validation_task'],
        )
        
    @crew
    def base_reasoner_crew(self) -> Crew:
        return Crew(
            agents=[self.base_reasoner()],
            tasks=[self.base_solving_task()],
            verbose=True,
        )
        
    @crew
    def selector_crew(self) -> Crew:
        return Crew(
            agents=[self.selector()],
            tasks=[self.selection_task()],
            verbose=True,
        )
    
    @crew
    def validator_crew(self) -> Crew:
        return Crew(
            agents=[self.validator()],
            tasks=[self.validation_task()],
            verbose=True,
        )
        

    @crew
    def crew(self) -> Crew:
        """Creates the MathCrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
