"""Build a Qdrant index from the corpus, two ways.

    --strategy sections   chunk on section boundaries, from corpus.db.
                          Every chunk carries act, section and tax year.
    --strategy naive      chunk the raw PDF by character count.
                          No section metadata, so answers cannot cite.

    python corpus-builder/scripts/index.py --strategy sections \
        --db data/corpus.db --act ITA-2025 \
        --qdrant http://localhost:6333 --collection ita_sections

The naive strategy is the baseline the section-aware one is measured
against. Keeping both is what turns a claim into a number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus import chunking, embedding, guidance, vectorstore  # noqa: E402


def summarise(strategy: str, chunks: list) -> None:
    lengths = sorted(len(c.text) for c in chunks)

    print(f"strategy    {strategy}")
    print(f"chunks      {len(chunks)}")
    print(
        f"length      min {lengths[0]}  "
        f"median {lengths[len(lengths) // 2]}  max {lengths[-1]}"
    )

    if strategy in ("sections", "guidance"):
        citable = len({c.payload["section_number"] for c in chunks})
        label = "sections" if strategy == "sections" else "pages"
        print(f"{label:<11} {citable} distinct, citable")
    else:
        print("sections    0 - naive chunks cannot cite a section")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy", choices=("naive", "sections", "guidance"), required=True
    )
    parser.add_argument("--act", default="ITA-2025")
    parser.add_argument("--pdf", type=Path, help="naive strategy input")
    parser.add_argument("--db", type=Path, help="sections strategy input")
    parser.add_argument(
        "--guidance",
        type=Path,
        default=Path("data/raw/guidance"),
        help="guidance strategy input directory",
    )
    parser.add_argument("--qdrant", default="data/index", help="URL or path")
    parser.add_argument("--collection")
    parser.add_argument("--size", type=int, default=1000, help="naive chunk size")
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument(
        "--dry-run", action="store_true", help="chunk only, no API calls"
    )
    args = parser.parse_args()

    if args.strategy == "guidance":
        chunks = guidance.from_directory(args.guidance)
    elif args.strategy == "sections":
        if not args.db:
            parser.error("--db is required for the sections strategy")
        chunks = chunking.from_sections(args.db, args.act)
    else:
        if not args.pdf:
            parser.error("--pdf is required for the naive strategy")
        chunks = chunking.from_pdf(args.pdf, args.act, args.size, args.overlap)

    if not chunks:
        parser.error(f"no chunks produced for {args.act}")

    summarise(args.strategy, chunks)

    if args.dry_run:
        print("\nsample chunk:\n")
        print(chunks[len(chunks) // 2].text[:600])
        return

    collection = args.collection or f"ita_{args.strategy}"

    print(f"\nembedding with {embedding.MODEL}...")
    vectors = embedding.embed_all([c.text for c in chunks])

    print(f"uploading to {args.qdrant}...")
    vectorstore.upload(vectorstore.connect(args.qdrant), collection, chunks, vectors)
    print(f"  uploaded {len(chunks)} points to {collection!r}")


if __name__ == "__main__":
    main()
