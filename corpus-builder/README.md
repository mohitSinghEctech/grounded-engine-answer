# corpus-builder

Local batch tooling that turns the published Acts into a Qdrant index.
**Never containerised** — none of this runs inside `llm-gateway` or
`tax-agent`, and none of it belongs in a service image. Keeping it out is
why `tax-agent` is 73 MB instead of ~400 MB.

## Pipeline

    data/raw/*.pdf  ──parse.py──▶  data/corpus.db  ──index.py──▶  Qdrant
                                                    search.py ──▶  inspect

## Commands

Run from the repository root.

    python corpus-builder/scripts/parse.py \
        --pdf data/raw/ita-2025.pdf --act ITA-2025 --db data/corpus.db
    python corpus-builder/scripts/parse.py \
        --pdf data/raw/ita-1961.PDF --act ITA-1961 --db data/corpus.db

    python corpus-builder/scripts/index.py --strategy sections \
        --db data/corpus.db --act ITA-2025 \
        --qdrant http://localhost:6333 --collection ita_sections

    python corpus-builder/scripts/search.py "deduction for life insurance premium" \
        --qdrant http://localhost:6333 --collection ita_sections --tax-year 2027

`index.py` replaces only the points for the act being indexed, so one
collection can hold both Acts.

## Dialects

The two Acts are structurally different documents and need different
detection rules, selected by `--act`:

| | ITA-2025 | ITA-1961 |
|---|---|---|
| Marginal note | previous line | same line as the number |
| Note terminator | full stop | `. - Body` or `. [Substituted by…` |
| Numbering | integer 1…536 | alphanumeric `80C`, `80-IA`, `115JD` |
| Sequence rule | exact successor | sort key `(number, suffix)` |
| Em dash | correct | extracts as `€` |
| Schedules | `SCHEDULE I` heading | no heading; numbering restarts |

## Environment

Reads `EMBEDDING_API_KEY`, `EMBEDDING_MODEL` and `EMBEDDING_DIMS`, falling
back to `OPENAI_API_KEY`. Sourcing `services/tax-agent/.env` is enough —
the index must be built with the same model the service queries with.
