"""The pipeline as a graph, so the path through it can branch.

`state.py` is what travels between nodes, `nodes.py` wraps each stage and
holds the routers that choose the next edge, and `build.py` wires them
together.

Why a graph at all: the straight-line pipeline can refuse early, but it
cannot go back. Retrying a generation, or searching again with a filter
dropped, means re-entering a stage that already ran - which a function
expresses as nested conditionals and a graph expresses as an edge.
"""
