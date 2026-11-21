from typing import Optional, Dict
from string import Template


class Prompt:
    def __init__(self, prompt_path: Optional[str] = None,
                 variables: Optional[Dict] = None) -> None:
        self.variables = variables if variables else {}
        self.p_path = prompt_path if prompt_path else None
        if self.p_path:
            with open(self.p_path, 'r') as f:
                self._prompt_template = f.read()
            self._prompt = self._build_prompt_with_template()
        else:
            self._prompt = ""

    def _render_prompt(self, **value: object) -> str:
        template = Template(self._prompt_template)
        try:
            return template.substitute(value).strip()
        except KeyError as e:
            raise KeyError(
                f"Varriable unknow in {self.p_path}: {e}\n"
                "Check the config file and prompt file"
            )

    def _build_prompt_with_template(self) -> str:
        return self._render_prompt(
            **self.variables
        )

    @property
    def prompt(self) -> str:
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: str) -> None:
        self._prompt = prompt
