import re
import subprocess
import json
from pathlib import Path

from dataclasses import asdict, dataclass
from jinja2 import Template

from local import LocalEnvironment
from model import ModelAdapter
from exceptions import (
    NonTerminatingException,
    FormatError,
    ExecutionTimeoutError,
    TerminatingException,
    Submitted,
    LimitsExceeded,
)


@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = """
[SYSTEM]
    You are a helpful assistant that can interact multiple times with a computer shell to solve fix issue in local github repository.
    This is an interactive process where you will think and issue ONE command, see its result, then think and issue your next command.

    FORMAT: 
        - Include a THOUGHT section before your command where you explain your reasoning process.
        - Format your response as shown in <format_example>.
            <format_example>
            THOUGHT: Your reasoning and analysis here

            ```bash
            your_command_here
            ```
            </format_example>
        - In the output, commands must be specified in a single bash code block:
            ```bash
            your_command_here
            ```
        - Failure to follow these rules will cause your response to be rejected.
        - Your response MUST include EXACTLY ONE bash code block
        - This bash block MUST contain EXACTLY ONE command (or a set of commands connected with && or ||)
        - If you include zero or multiple bash blocks, or no command at all, YOUR RESPONSE WILL FAIL
        - Do NOT try to run multiple independent commands in separate blocks in one response
        - Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
        - However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files
        - Always use non-interactive flags (-y, -f) for commands
        - Avoid interactive tools like vi, nano, or any that require user input

    ENVIRONMENT:
        - You have a full Linux shell environment
        - If a command isn't available, you can install it

    SUBMISSION:
        When you've completed your work (reading, editing, testing), and cannot make further progress 
        output following command:
        ```bash
        echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
        ```
        This command will submit your work.
        You cannot continue working (reading, editing, testing) in any way on this task after submitting.

    CONSTRAINTS:
        - Your task is specifically to make changes to non-test files in the current directory in order to fix the issue in a way that is general and consistent with the codebase.
            -- MODIFY: Regular source code files in {{working_dir}}
            -- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.), or any other files that are not part of the source code.
        - Don't commit or stash any changes to the repository.
        - Only use 'git diff' to check the changes you made.
"""

    instance_template: str = """
[USER]
    TASK: 
        {{task}}
    PROCESS:
        1. Analyze the codebase by finding and reading relevant files
        2. Create a script to reproduce the issue
        3. Edit the source code to resolve the issue
        4. Verify your fix works by running your script again
        5. Test edge cases to ensure your fix is robust
    Command Execution Rules
        You are operating in an environment where
        1. You write a single command
        2. The system executes that command in a subshell
        3. You see the result
        4. You write your next command
"""

    command_execution_template: str = """
[COMMAND_EXECUTION]
    ```bash
    {{command}}
    ```
"""

    action_observation_template: str = """
[OBSERVATION]
    <returncode>{{output.returncode}}</returncode>
    {% if output.output | length < 10000 -%}
    <output>
    {{ output.output -}}
    </output>
    {%- else -%}
    <warning>
    The output of your last command was too long.
    Please try a different command that produces less output.
    If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
    If you're using grep or find and it produced too much output, you can use a more selective search pattern.
    If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.
    </warning>
    {%- set elided_chars = output.output | length - 10000 -%}
    <output_head>
    {{ output.output[:5000] }}
    </output_head>
    <elided_chars>
    {{ elided_chars }} characters elided
    </elided_chars>
    <output_tail>
    {{ output.output[-5000:] }}
    </output_tail>
    {%- endif -%}
"""
    format_error_template: str = """
[ERROR_DURING_ACTION_EXECUTION]
    Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions.

    Please format your action in triple backticks as shown in <response_example>.

    <response_example>
    Here are some thoughts about why you want to perform the action.

    ```bash
    <action>
    ```
    </response_example>

    If you have completed your assignment, please consult the first message about how to
    submit your solution (you will not be able to continue working on this task after that)."""

    step_limit: int = 40


class DefaultAgent:
    """DefaultAgent implementation from check.py"""

    def __init__(
        self,
        model: ModelAdapter,
        env: LocalEnvironment,
    ):
        self.config = AgentConfig()
        self.messages: list[str] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.trajectory = {
            "prompts": [],
            "responses": [],
        }

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars()
        return Template(template).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, content: str):
        self.messages.append(content)

    def run(self, task: str):
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task}
        self.messages = []
        self.add_message(self.render_template(self.config.system_template))
        self.add_message(self.render_template(self.config.instance_template))

        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message(f"[ERROR] {str(e)}")
            except TerminatingException as e:
                self.add_message(f"[ERROR] {str(e)}")
                return 

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        response = self.query()
        observation = self.get_observation(response)
        return observation

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls:
            raise LimitsExceeded()
        response = self.model.query(self.messages)

        # Save result to check the trajectory
        self.trajectory["prompts"].append(response["prompt"])
        self.trajectory["responses"].append(response["content"])

        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(
            self.config.action_observation_template, output=output
        )
        self.add_message(observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(r"```bash\n(.*?)\n```", response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(
            self.render_template(self.config.format_error_template, actions=actions)
        )

    def execute_action(self, action: dict) -> dict:
        command = action["action"]

        # Automatically activate conda environment for Python commands
        if any(
            python_cmd in command.lower() for python_cmd in ["python", "pip", "conda"]
        ):
            command = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate testbed && {command}"

        # Add command execution template
        self.add_message(
            self.render_template(
                self.config.command_execution_template, command=command
            )
        )

        try:
            output = self.env.execute(command)
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(
                    self.config.format_error_template, action=action, output=output
                )
            )
        except TimeoutError:
            raise ExecutionTimeoutError(
                self.render_template(
                    self.config.format_error_template, action=action, output=""
                )
            )

        # Check if task is finished
        self.has_finished(output)
        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in lines[0].strip():
            final_output = "".join(lines[1:])
            raise Submitted(final_output)

    def create_diff(self) -> str:
        command = "git diff"
        try:
            output = self.env.execute(command)
            self.env.execute("git restore .")
            return output["output"]
        except Exception as e:
            print(f"Error creating diff: {e}")
            return ""

    def save_trajectory(self, path: Path):
        trajectory_data = []

        for i in range(len(self.trajectory["prompts"])):
            trajectory_data.append(
                {
                    "step": i,
                    "prompt": self.trajectory["prompts"][i],
                    "response": self.trajectory["responses"][i],
                }
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trajectory_data, indent=2))
        print(f"Saved trajectory to '{path}'")
