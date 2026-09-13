import re

# FY 2026-27 is stored as 2026. AY 2027-28 refers to FY 2026-27, so it maps
# to 2026 as well - the single most common source of off-by-one in Indian
# tax software.
_ASSESSMENT = re.compile(
    r"\b(?:a\.?\s?y\.?|assessment\s+year)\s*:?\s*(20\d{2})", re.IGNORECASE
)
_FINANCIAL = re.compile(
    r"\b(?:f\.?\s?y\.?|financial\s+year|tax\s+year|previous\s+year)\s*:?\s*(20\d{2})",
    re.IGNORECASE,
)
_BARE_RANGE = re.compile(r"\b(20\d{2})\s*[-–/]\s*\d{2,4}\b")

_MIN_YEAR = 1961
_MAX_YEAR = 2100


def _valid(year: int) -> int | None:
    return year if _MIN_YEAR <= year <= _MAX_YEAR else None


def extract_tax_year(question: str) -> int | None:
    """Find the tax year a question is about, as the starting calendar year.

    "deduction limit for FY 2026-27"  -> 2026
    "what changed in AY 2027-28"      -> 2026
    "rates for 2026-27"               -> 2026
    "what is section 123"             -> None
    """
    match = _ASSESSMENT.search(question)

    if match:
        return _valid(int(match.group(1)) - 1)

    match = _FINANCIAL.search(question)

    if match:
        return _valid(int(match.group(1)))

    match = _BARE_RANGE.search(question)

    if match:
        return _valid(int(match.group(1)))

    return None
