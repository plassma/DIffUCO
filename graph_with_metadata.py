import jraph
from flax import struct


@struct.dataclass
class GraphWithMeta:
    graph: jraph.GraphsTuple
    meta: dict = struct.field(pytree_node=False)
    