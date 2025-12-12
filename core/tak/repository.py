from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, TYPE_CHECKING
import logging
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from .tak import TAK
    from .raw_concept import RawConcept
    from .event import Event
    from .state import State
    from .trend import Trend
    from .context import Context
    from .pattern import Pattern

# Global singleton TAKRepository instance (set by mediator at startup)
_GLOBAL_TAK_REPO: Optional["TAKRepository"] = None

def get_tak_repository() -> "TAKRepository":
    """Access the global TAK repository."""
    if _GLOBAL_TAK_REPO is None:
        raise RuntimeError("TAK repository not initialized. Call set_tak_repository() first.")
    return _GLOBAL_TAK_REPO

def set_tak_repository(repo: "TAKRepository") -> None:
    """Set the global TAK repository (called once by mediator)."""
    global _GLOBAL_TAK_REPO
    _GLOBAL_TAK_REPO = repo


@dataclass
class TAKRepository:
    """
    Registry of all parsed TAK objects (raw-concepts, states, trends, etc.).
    Does not cache .df (computed on-demand to save memory).
    """
    taks: Dict[str, TAK] = field(default_factory=dict)  # {tak_name: TAK_instance}
    graph: Dict[str, Set[str]] = field(default_factory=dict)  # Dependency graph
    execution_order: List[str] = field(default_factory=list)  # Topological sort result

    def register(self, tak: TAK) -> None:
        """Register a TAK by name."""
        if tak.name in self.taks:
            raise ValueError(f"Duplicate TAK name: {tak.name}")
        self.taks[tak.name] = tak

    def get(self, name: str) -> Optional[TAK]:
        """Retrieve a TAK by name."""
        return self.taks.get(name)

    def finalize_repository(self) -> None:
        """
        Finalize the TAK repository after all TAKs are registered.
        Detects circular references and computes topological execution order.
        (Individual TAK validation already happens in each TAK's .parse() method.)
        """
        # 1) Build dependency graph and detect circular references
        self.graph = self._build_dependency_graph()  # Store as instance attribute
        self._detect_circular_references(self.graph)

        # 2) Compute topological sort order
        self.execution_order = self._topological_sort(self.graph)
        logger.info(f"TAK execution order: {self.execution_order}")

    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """
        Build a dependency graph: {tak_name: set_of_dependent_tak_names}
        A TAK depends on all TAKs it references (derived-from, parameters, clippers, etc.)
        
        Returns deduplicated dependency sets (Pattern references same TAK twice → appears once).
        """
        graph: Dict[str, Set[str]] = {name: set() for name in self.taks}

        for tak_name, tak in self.taks.items():
            dependencies = self._extract_dependencies(tak)
            graph[tak_name] = dependencies  # set() automatically deduplicates

        return graph

    def _extract_dependencies(self, tak: TAK) -> Set[str]:
        """Extract all TAK names that this TAK depends on (deduplicated)."""
        # Import here to avoid circular imports
        from .raw_concept import RawConcept, ParameterizedRawConcept
        from .event import Event
        from .state import State
        from .trend import Trend
        from .context import Context
        from .pattern import Pattern

        deps = set()

        # Raw Concept: no dependencies (data source)
        if isinstance(tak, RawConcept) and not isinstance(tak, ParameterizedRawConcept):
            return deps
        
        # ParameterizedRawConcept: depends on derived-from + parameters (check BEFORE RawConcept!)
        if isinstance(tak, ParameterizedRawConcept):
            deps.add(tak.derived_from)
            for param in tak.parameters:
                deps.add(param["name"])
            return deps

        # Event: depends on derived-from raw-concepts
        if isinstance(tak, Event):
            for df in tak.derived_from:
                deps.add(df["name"])
            return deps

        # State: depends on derived-from (raw-concept or event)
        if isinstance(tak, State):
            deps.add(tak.derived_from)
            return deps

        # Trend: depends on derived-from raw-concept
        if isinstance(tak, Trend):
            deps.add(tak.derived_from)
            return deps

        # Context: depends on derived-from + clippers
        if isinstance(tak, Context):
            for df in tak.derived_from:
                deps.add(df["name"])
            for clipper in tak.clippers:
                deps.add(clipper["name"])
            return deps

        # Pattern: depends on derived-from + parameters TAKs
        if isinstance(tak, Pattern):
            for df in tak.derived_from:
                deps.add(df["name"])
            for param in tak.parameters:
                deps.add(param["name"])
            return deps

        return deps

    def _detect_circular_references(self, graph: Dict[str, Set[str]]) -> None:
        """
        Detect circular references using DFS.
        Raises ValueError if a cycle is found.
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in self.taks:
                    # Neighbor not in repository (missing TAK)
                    raise ValueError(f"TAK '{node}' references missing TAK '{neighbor}'")

                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle_path = path[cycle_start:] + [neighbor]
                    raise ValueError(
                        f"Circular reference detected: {' → '.join(cycle_path)}"
                    )

            rec_stack.discard(node)

        # Run DFS from each unvisited node
        for tak_name in self.taks:
            if tak_name not in visited:
                dfs(tak_name, [])

    def _topological_sort(self, graph: Dict[str, Set[str]]) -> List[str]:
        """
        Compute topological sort order using Kahn's algorithm.
        Returns list of TAK names in execution order (dependencies first).
        
        graph: {tak_name: set_of_dependencies_it_needs}
        We need to reverse this to get {tak_name: set_of_dependents_that_need_it}
        """
        # 1. Build reverse graph: {dependency: set_of_taks_that_need_it}
        reverse_graph: Dict[str, Set[str]] = {name: set() for name in self.taks}
        for tak_name, deps in graph.items():
            for dep in deps:
                reverse_graph[dep].add(tak_name)
        
        # 2. Calculate in-degree (number of dependencies each TAK has)
        in_degree = {name: len(deps) for name, deps in graph.items()}
        
        # 3. Initialize queue with TAKs that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            # Sort queue by TAK family for predictable ordering
            queue.sort(key=lambda name: (self._family_priority(self.taks[name].family), name))
            node = queue.pop(0)
            order.append(node)
            
            # For each TAK that depends on this node, decrement its in-degree
            for dependent in reverse_graph[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(order) != len(self.taks):
            # Should never happen if circular reference detection works
            missing = set(self.taks.keys()) - set(order)
            raise RuntimeError(f"Topological sort failed. Missing TAKs: {missing}")
        
        return order

    @staticmethod
    def _family_priority(family: str) -> int:
        """Return priority for execution order (lower = earlier)."""
        priority_map = {
            "raw-concept": 0,
            "event": 1,
            "state": 2,
            "trend": 3,
            "context": 4,
            "pattern": 5,
            "local-pattern": 5,
            "global-pattern": 6,
        }
        return priority_map.get(family, 99)