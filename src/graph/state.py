from typing_extensions import Annotated, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage
import json


def merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


class AgentState(TypedDict):
    """统一状态，在所有 Agent 间传递"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict, merge_dicts]
    metadata: Annotated[dict, merge_dicts]


def show_agent_reasoning(output, agent_name: str):
    """格式化打印 Agent 的输出信号"""
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")

    def convert(obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        else:
            return str(obj)

    if isinstance(output, (dict, list)):
        print(json.dumps(convert(output), indent=2, ensure_ascii=False))
    else:
        try:
            parsed = json.loads(output)
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        except (json.JSONDecodeError, TypeError):
            print(output)

    print('=' * 48)
