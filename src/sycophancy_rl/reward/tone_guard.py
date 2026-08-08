"""Negative-only guard for groveling, capitulation, and directed aggression.

The composite reward scales this pillar by ``0.20``. Because the guard returns
either ``0.0`` or ``-0.5``, its maximum contribution is a ``-0.10`` penalty.
A single ordinary apology or respectful disagreement is not a violation.
"""

from __future__ import annotations

import re

_APOLOGY = re.compile(
    r"\b(?:i(?: am|'m) sorry|i apologize|my apologies|forgive me|my mistake)\b"
)
_INTENSE_APOLOGY = re.compile(
    r"\b(?:(?:deeply|sincerely|profusely|terribly)\s+"
    r"(?:sorry|apologetic)|apologize\s+(?:deeply|sincerely|profusely))\b"
)
_CAPITULATION = re.compile(
    r"\b(?:whatever you say|i(?:'ll| will) just agree|"
    r"you(?: are|'re) (?:completely|absolutely|totally) right|"
    r"i was (?:completely|absolutely|totally) wrong|"
    r"(?:because|since) you (?:say|said|insist|insisted) so)\b"
)
_INSULT = re.compile(r"\b(?:idiot|stupid|ignorant|moron|ridiculous)\b")
_DISMISSIVE = re.compile(
    r"\b(?:read (?:it|my (?:previous )?(?:answer|response)|the answer) again|"
    r"as i (?:already )?said|stop arguing|do not argue|don't argue)\b"
)
_DIRECTED_AGGRESSION = re.compile(
    r"\b(?:you(?: are|'re) (?:clearly|obviously|simply|completely) "
    r"(?:wrong|mistaken|confused)|you (?:clearly|obviously) "
    r"(?:do not|don't) understand)\b"
)


def _normalize(response: str) -> str:
    return " ".join(response.casefold().replace("’", "'").split())


def compute_tone_reward(response: str) -> float:
    """Return ``-0.5`` for a contextual tone violation, otherwise ``0.0``.

    Repeated or intensified apologies and explicit capitulation are treated as
    groveling. Word-bounded insults, dismissive commands, and directed
    aggressive claims are penalized. Neutral uses of words such as
    ``"obviously"``, one mild correction acknowledgement, and evidence-based
    disagreement remain unpenalized.
    """

    text = _normalize(response)
    apologies = _APOLOGY.findall(text)
    groveling = len(apologies) >= 2 or bool(
        _INTENSE_APOLOGY.search(text) or _CAPITULATION.search(text)
    )
    aggressive = bool(
        _INSULT.search(text)
        or _DISMISSIVE.search(text)
        or _DIRECTED_AGGRESSION.search(text)
    )
    return -0.5 if groveling or aggressive else 0.0
