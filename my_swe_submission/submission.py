from swebase import SWEBase

from local import LocalEnvironment
from agent import DefaultAgent
from model import ModelAdapter


class SWE(SWEBase):
    """🤖 Enhanced Mini-SWE-Agent with sophisticated logic from check.py"""

    def __init__(self):
        super().__init__()
        self.step_count = 0
        self.max_steps = 15  # Reduced to be more efficient
        self.timeout = 30

    def __call__(self, repo_location: str, issue_description: str) -> str:
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
            diff = agent.create_diff(repo_location)

            return diff

        except Exception as e:
            print(f"❌ Error: {e}")
            return ""


# Enhanced testing section
if __name__ == "__main__":
    from dotenv import load_dotenv
    import pickle as pkl

    from coding.tasks.swe import SWEBenchTask
    from coding.schemas.context import Context
    from coding.datasets.swefull import SWEFullDataset

    dataset = SWEFullDataset()
    context_dict = dataset.get(n=1)
    context = Context(**context_dict)
    task = SWEBenchTask(llm=None, context=context, use_remote=False)

    load_dotenv()

    # print(task.row["instance_id"])
    # with open(f"./problems/task_{task.row['instance_id']}.pkl", "wb") as f:
    #     pkl.dump(task, f)

    # with open("./problems/task_django__django-11239.pkl", "rb") as f:
    #     task = pkl.load(f)

    swe = SWE()
    response = swe(repo_location=task.repo.path, issue_description=task.query)

    score = task.score(response)
    print(f"\n🎯 Final Score: {score}")
