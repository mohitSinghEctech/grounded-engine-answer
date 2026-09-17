"""The work the pipeline does, separated from the order it happens in.

`steps.py` holds one function per stage - retrieve, prompt, generate,
verify. None of them decide what runs next.

Two orchestrators call those same functions: the straight line in
`routers/ask.py`, and the LangGraph graph in `app/graph/`. Because the
stages live here and not in either orchestrator, switching between them
cannot change what a stage does - only the order and the branching. That
is what makes the two comparable in the eval harness.
"""
