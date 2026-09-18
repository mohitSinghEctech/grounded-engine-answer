"""The tool-calling loop: the model chooses what runs, within bounds.

`tools.py` is what the model may ask for - schemas it reads, handlers that
run. `loop.py` is the turn-taking, the three exits and the budgets.

The distinction from `app/graph/`: the graph routes along edges *your*
code chooses, while here the model chooses. Both keep the same grounding
guarantee - every citation is checked against provisions a tool actually
returned.
"""
