import jraph
from flax import struct


@struct.dataclass
class GraphWithMeta:
    graph: jraph.GraphsTuple
    meta: dict = struct.field(pytree_node=False)

    def __getattr__(self, name):
        """Delegate attribute lookups to the underlying graph."""
        if name == "meta":
            raise AttributeError(f"{type(self).__name__} has no attribute {name}")
        try:
            graph = object.__getattribute__(self, "graph")
        except AttributeError as exc:  # graph not yet set during unpickling
            raise AttributeError(name) from exc
        return getattr(graph, name)

    def __dir__(self):
        """Expose graph attributes in auto-completion and introspection."""
        try:
            graph = object.__getattribute__(self, "graph")
        except AttributeError:
            graph = None
        graph_dir = set(dir(graph)) if graph is not None else set()
        return sorted(set(super().__dir__()) | graph_dir)
    
