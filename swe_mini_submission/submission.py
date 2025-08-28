from pathlib import Path
from datetime import datetime

from swebase import SWEBase

from local import LocalEnvironment
from agent import DefaultAgent
from model import ModelAdapter


class SWE(SWEBase):
    """🤖 Enhanced Mini-SWE-Agent with sophisticated logic from check.py"""

    def __init__(self):
        super().__init__()

    def __call__(self, repo_location: str, issue_description: str, instance_id: str = "") -> tuple[str, int]:
        try:
            # Create model adapter
            model = ModelAdapter(self.llm, "openai/gpt-5-mini")

            # Create environment
            env = LocalEnvironment(cwd=repo_location)

            # Create agent with configuration
            agent = DefaultAgent(model, env)

            # Run the agent
            agent.run(issue_description)

            # Create patch from the repository changes
            diff = agent.create_diff()

            # Save trajectory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            agent.save_trajectory(Path(f"./trajectories/trajectory_opt_{instance_id}_{timestamp}.json"))

            return diff

        except Exception as e:
            print(f"❌ Error: {e}")
            return "", 0


# Enhanced testing section
if __name__ == "__main__":
    from dotenv import load_dotenv
    # from coding.tasks.swe import SWEBenchTask
    # from coding.schemas.context import Context
    # from coding.datasets.swefull import SWEFullDataset
    import pickle as pkl
    import os

    load_dotenv()

    # n_problems = 10

    # for index in range(n_problems):
    #     dataset = SWEFullDataset()
    #     context_dict = dataset.get(n=1)
    #     context = Context(**context_dict)
    #     task = SWEBenchTask(llm=None, context=context, use_remote=False)

    #     print(f"Task: {task.row['instance_id']}")
    #     with open(f"./problems/task_{task.row['instance_id']}.pkl", "wb") as f:
    #         pkl.dump(task, f)

    folder = "./problems"
    total_score = 0
    not_solved = []
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), "rb") as f:
            task = pkl.load(f)

        print(f"Task: {task.row['instance_id']}")

        swe = SWE()
        response = swe(repo_location=task.repo.path, issue_description=task.query, instance_id=task.row["instance_id"])

        score = task.score(response)
        print(f"\n🎯 Final Score: {score}")
        total_score += score
        if score == 0:
            not_solved.append(task.row["instance_id"])

    print(f"\n🎯 Total Score: {total_score}")
    print(f"Not solved: {not_solved}")