from swebase import LLMClient


class ModelAdapter:
    """Adapter to make SWEBase LLM compatible with agent interface"""

    def __init__(self, llm_client: LLMClient, model_name: str):
        self.llm_client = llm_client
        self.cost = 0.0
        self.n_calls = 0
        self.model_name = model_name

    def query(self, messages: list[dict[str, str]]) -> dict:
        # Convert messages to the format expected by SWEBase
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

        # Create a single prompt from all messages
        prompt_parts = []
        for msg in formatted_messages:
            if msg["role"] == "system":
                prompt_parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")

        prompt = "\n\n".join(prompt_parts)

        try:
            response, tokens = self.llm_client(prompt, self.model_name, 0.0)
            self.n_calls += 1
            self.cost += tokens * 0.00001

            return {"content": response or ""}
        except Exception as e:
            print(f"❌ LLM Query Failed: {e}")
            return {"content": ""}
