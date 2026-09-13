"""Turn published Acts into a searchable vector index.

The pipeline has four stages, each in its own module:

    pdf.py        PDF file      -> clean text with page offsets
    dialects/     clean text    -> sections (one module per Act)
    storage.py    sections      -> SQLite, and back
    chunking.py   sections      -> chunks small enough to embed
    embedding.py  chunks        -> vectors
    vectorstore.py vectors      -> Qdrant, and search over it

Nothing here runs inside a service container. It is batch tooling that
produces an index; the services only ever read what it writes.
"""
