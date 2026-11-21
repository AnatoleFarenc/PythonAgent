from .Orchestrateur import Orchestrateur
from pathlib import Path

def main() -> None:
    orchestrateur = Orchestrateur(
        model_name="baidu/ernie-5.5-flash-preview:free",
        provider_url="https://openrouter.ai/api/v1/",
    )
    result = orchestrateur.complete_task("give me a code in \"conifg.py\"")
    Path("result.json").write_text(result.model_dump_json(indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()
