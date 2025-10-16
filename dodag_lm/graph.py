"""Utilities for working with DAG-structured language contexts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Tuple


@dataclass
class DodagNode:
    """A node describing a word token and its outgoing edges."""

    token_id: int
    children: List[int] = field(default_factory=list)

    def add_child(self, child_id: int) -> None:
        if child_id not in self.children:
            self.children.append(child_id)


class DodagGraph:
    """A light-weight container for parent/child relationships."""

    def __init__(self, root_token: int) -> None:
        self.root_token = root_token
        self.nodes: Dict[int, DodagNode] = {root_token: DodagNode(root_token)}
        self.edge_list: List[Tuple[int, int]] = []

    def get_or_create(self, token_id: int) -> DodagNode:
        if token_id not in self.nodes:
            self.nodes[token_id] = DodagNode(token_id)
        return self.nodes[token_id]

    def add_edge(self, parent_id: int, child_id: int) -> None:
        parent = self.get_or_create(parent_id)
        parent.add_child(child_id)
        self.get_or_create(child_id)
        self.edge_list.append((parent_id, child_id))

    def edges(self) -> Iterable[Tuple[int, int]]:
        return iter(self.edge_list)

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[DodagNode]:
        return iter(self.nodes.values())
