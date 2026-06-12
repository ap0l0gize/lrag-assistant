from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ContextItem:
    id: str
    source: Literal["graph", "vector"]
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContextItem":
        return cls(
            id=data["id"],
            source=data["source"],
            content=data["content"],
            metadata=data.get("metadata", {}),
        )
