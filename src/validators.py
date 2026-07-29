"""Format validation for high-risk OCR fields (UK postcodes, phones, emails).

Digits and proper nouns are where a vision model fails first: they carry no
linguistic context to fall back on, so a misread `B` -> `8` produces something
that looks perfectly plausible. On a safeguarding form those are exactly the
fields that matter — a wrong postcode or phone number is worse than a wrong
adjective.

These fields are *format-constrained*, so a chunk of that error class can be
caught with no reference text at all. Three checks run here:

  1. Structural  — does the token match the field's grammar? A UK postcode
     cannot have a digit in its final two positions, so `TS25 28Q` is provably
     wrong. When exactly one OCR confusion substitution repairs it, that fix is
     unambiguous and is offered as a suggestion.
  2. Consistency — the same entity written two ways in one document means at
     least one reading is wrong. `TS25 28Q` in section 1 and `TS2S 2SQ` in
     section 2 disagree, so one is corrupt even though the second is
     structurally legal.
  3. Plausibility — an 01xxx phone number has 11 digits. Anything else is a
     dropped or invented character.

WHAT THIS CANNOT DO: it validates form, not truth. `01422 065770` is a
perfectly well-formed UK number that happens to be Halifax rather than the
Teesside `01642` the document actually says — no format rule catches that. Only
a lookup against known-good values would, and that is document-specific. Treat
a clean report as "no detectable format errors", never as "no errors".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# Final two letters of a UK postcode never use C, I, K, M, O or V.
_INWARD_LETTERS = "ABD-HJLNP-UW-Z"
_POSTCODE_RE = re.compile(
    rf"^[A-Z]{{1,2}}[0-9][A-Z0-9]? ?[0-9][{_INWARD_LETTERS}]{{2}}$")

# Anything shaped enough like a postcode to be worth checking, including the
# malformed ones we want to flag (that is the point — the strict pattern above
# would skip exactly the tokens that are broken).
_POSTCODE_CANDIDATE_RE = re.compile(r"\b[A-Z0-9]{2,4} ?[0-9A-Z]{3}\b")

# Glyph pairs a vision model confuses. Bidirectional: read as digit->letters
# for slots that must hold a letter, and inverted for slots that must hold a
# digit.
_DIGIT_TO_LETTER = {
    "0": "ODQ", "1": "IL", "2": "Z", "4": "A",
    "5": "S", "6": "G", "8": "B", "9": "G",
}
_LETTER_TO_DIGIT: dict[str, str] = {}
for _digit, _letters in _DIGIT_TO_LETTER.items():
    for _letter in _letters:
        _LETTER_TO_DIGIT.setdefault(_letter, "")
        _LETTER_TO_DIGIT[_letter] += _digit

_PHONE_RE = re.compile(r"\b0[12378]\d{2,4}[ -]?\d{3,7}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]*\w\.[A-Za-z]{2,}\b")
_EMAIL_CANDIDATE_RE = re.compile(r"\S*@\S*")


class Severity(str, Enum):
    ERROR = "error"      # provably malformed — cannot be correct as written
    WARNING = "warning"  # well-formed but suspicious (disagrees with a sibling)


@dataclass
class Finding:
    field: str           # "postcode" | "phone" | "email"
    value: str           # the token as transcribed
    severity: Severity
    message: str
    suggestion: str | None = None  # set only when the repair is unambiguous


def _repair_postcode(token: str) -> str | None:
    """Return the unique confusion-repair of a malformed postcode, if unique.

    Each position in a postcode is constrained to a digit or a letter. Where the
    transcribed glyph has the wrong type, substitute its confusion partners and
    keep any candidate that satisfies the full pattern. Returns the fix only
    when exactly one candidate survives — an ambiguous repair is not a repair.
    """
    compact = token.replace(" ", "")
    if not 5 <= len(compact) <= 7:
        return None

    outward, inward = compact[:-3], compact[-3:]
    # Inward code is rigid: digit, letter, letter.
    slots = [_LETTER_TO_DIGIT, _DIGIT_TO_LETTER, _DIGIT_TO_LETTER]

    options: list[str] = []
    for char, table in zip(inward, slots):
        wants_digit = table is _LETTER_TO_DIGIT
        if char.isdigit() == wants_digit:
            options.append(char)          # already the right type
        else:
            options.append(table.get(char, ""))
        if not options[-1]:
            return None                   # no plausible substitution

    candidates = {
        f"{outward} {a}{b}{c}"
        for a in options[0] for b in options[1] for c in options[2]
        if _POSTCODE_RE.match(f"{outward} {a}{b}{c}")
    }
    return candidates.pop() if len(candidates) == 1 else None


def _normalize_postcode(value: str) -> str:
    return value.replace(" ", "").upper()


def _is_postcode_shaped(token: str) -> bool:
    """Reject tokens that merely resemble a postcode.

    Running against the real form surfaced three classes of false positive:
    prose ("ring the police ON 999", "CALL 101"), and internal references
    (Datix "W12345"). Two structural facts separate them from real postcodes:
    the outward code always contains a digit, and the inward code always ends
    in two letters.
    """
    outward, _, inward = token.partition(" ")
    if not inward:
        outward, inward = token[:-3], token[-3:]
    return (any(ch.isdigit() for ch in outward)
            and any(ch.isalpha() for ch in outward)
            and any(ch.isalpha() for ch in inward))


def _near_duplicates(codes: list[str], max_distance: int = 2) -> list[set[str]]:
    """Group postcodes within `max_distance` edits of each other.

    Two postcodes that differ by one or two characters in a document that
    repeats the same address are far more likely to be one postcode read twice
    than two genuinely different ones. Structural validation misses this class
    entirely: `TS2S 2SQ` breaks no rule, it is simply not what the page says.
    """
    from .evaluation import _levenshtein

    groups: list[set[str]] = []
    for code in codes:
        for group in groups:
            if any(_levenshtein(code, other) <= max_distance for other in group):
                group.add(code)
                break
        else:
            groups.append({code})
    return [g for g in groups if len(g) > 1]


def _check_postcodes(text: str) -> list[Finding]:
    findings: list[Finding] = []
    well_formed: list[str] = []

    for match in _POSTCODE_CANDIDATE_RE.finditer(text.upper()):
        token = match.group(0)
        if not _is_postcode_shaped(token):
            continue
        if _POSTCODE_RE.match(token):
            well_formed.append(_normalize_postcode(token))
            continue
        suggestion = _repair_postcode(token)
        if suggestion:
            well_formed.append(_normalize_postcode(suggestion))
        findings.append(Finding(
            field="postcode", value=token, severity=Severity.ERROR,
            message="not a valid UK postcode — the final two characters must be "
                    "letters (excluding C, I, K, M, O, V)",
            suggestion=suggestion,
        ))

    # Consistency: near-identical postcodes mean at least one reading is wrong.
    for group in _near_duplicates(sorted(set(well_formed))):
        findings.append(Finding(
            field="postcode", value=" / ".join(sorted(group)),
            severity=Severity.WARNING,
            message="these postcodes differ by only 1-2 characters — likely one "
                    "address misread two different ways; check against the page",
        ))
    return findings


_TERM_RE = re.compile(r"\b[A-Z][a-z]{4,}\b")


def _check_terms(text: str) -> list[Finding]:
    """Flag a proper noun that contradicts its own spelling elsewhere.

    Place and organisation names repeat across a form, so a name spelled two
    ways within one document has one wrong spelling — `Harlepool` against
    `Hartlepool`. Two design choices make this safe enough to auto-apply:

    * Occurrences are counted case-insensitively as substrings, so email
      domains vote. That matters: on the test form both spellings appeared 3
      times as words, and only `hartlepool.gov.uk` broke the tie 6-3.
    * A pair where the shorter term is a prefix of the longer is skipped. That
      is inflection, not misreading — otherwise `Concerns` "corrects" to
      `Concern` and the fix corrupts correct text.

    A clear majority (2x) is reported as an error with a suggestion; a near-tie
    is only a warning, because the document gives no evidence which is right.
    """
    from .evaluation import _levenshtein

    lowered = text.lower()
    terms = {m.group(0) for m in _TERM_RE.finditer(text)}
    counts = {t: lowered.count(t.lower()) for t in terms}

    findings: list[Finding] = []
    for term, other in ((a, b) for a in sorted(terms) for b in sorted(terms) if a < b):
        if _levenshtein(term, other) != 1:
            continue
        lo, hi = sorted((term, other), key=len)
        if hi.lower().startswith(lo.lower()):
            continue  # plural/inflection, not a misread

        n_term, n_other = counts[term], counts[other]
        loser, winner = ((term, other) if n_term < n_other else (other, term))
        if max(n_term, n_other) >= 2 * min(n_term, n_other):
            findings.append(Finding(
                field="term", value=loser, severity=Severity.ERROR,
                message=f"'{winner}' appears {counts[winner]}x against "
                        f"{counts[loser]}x for this spelling, differing by one "
                        f"character — the minority spelling is a misread",
                suggestion=winner,
            ))
        else:
            findings.append(Finding(
                field="term", value=f"{term} / {other}",
                severity=Severity.WARNING,
                message=f"one name spelled two ways ({n_term}x vs {n_other}x) — "
                        f"one is a misread, but the document does not say which",
            ))
    return findings


def _check_phones(text: str) -> list[Finding]:
    """Length-check UK numbers.

    Geographic (01/02) and non-geographic (03/07/08) numbers are 11 digits.
    A 10-digit form is legal only for a short list of legacy 01 areas, so it is
    reported as a warning rather than an error — on a scanned form it is much
    more often a dropped digit than a genuine Brampton number.
    """
    findings: list[Finding] = []
    for match in _PHONE_RE.finditer(text):
        token = match.group(0)
        digits = re.sub(r"\D", "", token)
        if len(digits) == 11:
            continue
        if len(digits) == 10:
            findings.append(Finding(
                field="phone", value=token, severity=Severity.WARNING,
                message="10 digits — nearly all UK numbers have 11, so a digit "
                        "was probably dropped in transcription",
            ))
            continue
        findings.append(Finding(
            field="phone", value=token, severity=Severity.ERROR,
            message=f"UK numbers have 11 digits; this has {len(digits)} — "
                    f"a character was dropped or invented",
        ))
    return findings


def _check_emails(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for match in _EMAIL_CANDIDATE_RE.finditer(text):
        token = match.group(0).strip(".,;:()[]<>")
        if not token or token.count("@") != 1:
            continue
        if not _EMAIL_RE.match(token):
            findings.append(Finding(
                field="email", value=token, severity=Severity.ERROR,
                message="malformed address — missing domain, missing TLD, or "
                        "stray characters",
            ))
    return findings


def validate(text: str) -> list[Finding]:
    """Run every format check over a transcription, errors first."""
    findings = (_check_postcodes(text) + _check_phones(text)
                + _check_emails(text) + _check_terms(text))
    return sorted(findings, key=lambda f: f.severity is not Severity.ERROR)


def apply_fixes(text: str) -> tuple[str, list[Finding]]:
    """Apply every unambiguous repair, returning the corrected text.

    Only findings carrying a `suggestion` are applied — those are the ones
    where exactly one candidate satisfied the field's grammar, so there is
    nothing to choose between. Findings without a suggestion (occlusion,
    inconsistent siblings, wrong-but-well-formed values) are returned
    unapplied for a human to resolve.
    """
    applied: list[Finding] = []
    for finding in validate(text):
        if not finding.suggestion:
            continue
        # Postcode tokens were upper-cased during detection; match the page.
        pattern = re.compile(re.escape(finding.value), re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(finding.suggestion, text)
            applied.append(finding)
    return text, applied


def format_report(findings: list[Finding]) -> str:
    """Human-readable validation report for the terminal."""
    if not findings:
        return ("Format validation: no detectable errors.\n"
                "  (checks form, not truth — a well-formed wrong value still passes)")

    errors = sum(1 for f in findings if f.severity is Severity.ERROR)
    lines = [f"Format validation: {errors} error(s), "
             f"{len(findings) - errors} warning(s)", "-" * 72]
    for f in findings:
        mark = "ERROR  " if f.severity is Severity.ERROR else "WARNING"
        lines.append(f"  [{mark}] {f.field}: {f.value}")
        lines.append(f"            {f.message}")
        if f.suggestion:
            lines.append(f"            -> did you mean: {f.suggestion}")
    return "\n".join(lines)
