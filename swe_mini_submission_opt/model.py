from swebase import LLMClient


class ModelAdapter:
    """Adapter to make SWEBase LLM compatible with agent interface"""

    def __init__(self, llm_client: LLMClient, model_name: str):
        self.llm_client = llm_client
        self.n_calls = 0
        self.model_name = model_name

    def query(self, messages: list[str]) -> dict:
        prompt = "\n\n".join(messages)

        try:
            response, _ = self.llm_client(prompt, self.model_name, 0.0)
            self.n_calls += 1

            return {"content": response or "", "prompt": prompt}
        except Exception as e:
            print(f"❌ LLM Query Failed: {e}")
            return {"content": ""}
