# LangGraph state definition
from typing import TypedDict, List, Annotated, Sequence
from langchain_core.messages import AnyMessage, BaseMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], None]
