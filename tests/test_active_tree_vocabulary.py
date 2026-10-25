"""Active implementation vocabulary stays aligned with the current design."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("src", "tests", "scripts")
SCAN_FILES = ("README.md",)


def _blocked_tokens() -> tuple[str, ...]:
    return (
        "ga" + "te",
        "admis" + "sion",
        "sha" + "dow",
        "ga" + "ted",
        "no_" + "ga" + "te",
        "admit" + "ted",
        "candi" + "date",
        "integra" + "tor",
    )


def _iter_text_files():
    for rel in SCAN_FILES:
        yield ROOT / rel
    for rel_root in SCAN_ROOTS:
        for path in (ROOT / rel_root).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md"}:
                yield path


def test_active_tree_uses_current_reflection_memory_vocabulary():
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, _blocked_tokens())) + r")\b", re.I)
    hits: list[str] = []
    for path in _iter_text_files():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert hits == []


#: Prose the reader sees. The code keeps ``arm`` in identifiers and JSON fields, so the
#: check above cannot carry these: it scans the tree where those names are correct.
PROSE_FILES = ("README.md",)
PROSE_GLOBS = ()

#: Renamed while writing up: an evaluation ``arm`` is a run or a learned memory, and a
#: scorecard ``cell`` is a record. Both were removed from the thesis by hand and nothing
#: stopped them returning; three forward-run records written afterwards used both again.
RETIRED_PROSE = ("arm", "arms", "cell", "cells")

#: Text a reader sees inside a figure. It is drawn from a script, so the prose check
#: above never looked at it, and the architecture figure still read "per (expert,
#: state) cell" while every sentence around it said record.
FIGURE_TEXT_FILES = ("scripts/architecture_figures.py",)


def test_figure_labels_use_the_thesis_vocabulary():
    pattern = re.compile(r"\b(" + "|".join(RETIRED_PROSE) + r")\b", re.I)
    hits: list[str] = []
    for rel in FIGURE_TEXT_FILES:
        path = ROOT / rel
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            # only the quoted label text is shown to a reader
            for label in re.findall(r'"([^"]*)"|\'([^\']*)\'', line):
                text = label[0] or label[1]
                if pattern.search(text):
                    hits.append(f"{rel}:{lineno}: {text[:80]}")
    assert hits == []

#: Sentences whose only work is to score the contribution down before a reader does.
#: "Both halves are borrowed" and "Three of the five components are existing practice"
#: were removed by hand on 2026-08-13, and "the systems it borrows from" grew back in
#: the same section two passes later. A citation says where a piece came from and stays;
#: a tally of how much is not new does not.
SELF_DOWNGRADE = (
    r"borrow(?:s|ed|ing)? from",
    r"(?:both|all) (?:halves|parts) are borrowed",
    r"(?:are|is) existing practice",
    r"nothing (?:here )?is new",
    r"(?:only|merely) an? (?:incremental|small) (?:step|contribution)",
    r"makes? no claim to novelty",
    r"none of (?:this|these) (?:is|are) novel",
)


def test_prose_does_not_tally_what_is_borrowed():
    """Attribution belongs inside the sentence that gives the mechanism, not in a
    count of how much of the design came from elsewhere."""
    pattern = re.compile(r"(" + "|".join(SELF_DOWNGRADE) + r")", re.I)
    hits: list[str] = []
    for rel in PROSE_FILES:
        path = ROOT / rel
        for lineno, line in enumerate(_prose(path).splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()[:110]}")
    assert hits == []


#: The other direction, and the easier one to write by accident. A sentence that tells
#: the reader a choice is hard, strong, careful or principled is arguing the work's
#: merit before anyone has questioned it; removing it costs nothing a reader needs.
#: "Equal weighting is a demanding benchmark to set" was added and removed on
#: 2026-08-13, in the same pass that removed the borrowed tally.
SELF_DEFENCE = (
    r"\b(?:demanding|strong|hard|tough|conservative|stringent) benchmark\b",
    r"\bhard to beat\b",
    r"\bdeliberately conservative\b",
    r"\berrs? on the side of\b",
    r"\bmore than fair\b",
    r"\bsets? a high bar\b",
    r"\b(?:principled|rigorous|careful|defensible|robust) (?:choice|design|procedure)\b",
    r"\bguards? against\b",
    r"\bprotects? (?:the|against|it|a)\b",
    r"\bsafeguards?\b",
    r"\bto be safe\b",
)


def test_prose_does_not_argue_its_own_merit():
    """A specification says what the procedure is. A defence says it is respectable."""
    pattern = re.compile(r"(" + "|".join(SELF_DEFENCE) + r")", re.I)
    hits: list[str] = []
    for rel in PROSE_FILES:
        path = ROOT / rel
        for lineno, line in enumerate(_prose(path).splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()[:110]}")
    assert hits == []


#: A cleft puts a whole clause in front of the verb to announce the point that follows
#: ("What separates the two learners is the spread"). It reads as emphasis and costs
#: three or four words over the plain form. Swept to zero on 2026-08-14.
CLEFT = (
    r"\bWhat (?:the|this|that|a|an|its|their|separates|makes|matters)\b[^.]{0,60}?\bis\b",
    r"\bis what [a-z]+(?:s|ed)\b",
    r"\bIt is (?:the|this|that|a|an)\b[^.]{0,40}?\bthat\b",
)


def test_prose_does_not_pad_with_cleft_sentences():
    """The plain form says the same thing sooner."""
    pattern = re.compile(r"(" + "|".join(CLEFT) + r")")
    hits: list[str] = []
    for rel in PROSE_FILES:
        path = ROOT / rel
        text = _prose(path)
        # Captions name their own exhibit ("What the memory changed on the six") and a
        # definition may legitimately read "X is what a call earned".
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("\\caption") or "\\cdot r$ is what" in line:
                continue
            if pattern.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()[:110]}")
    assert hits == []


def _prose(path: Path) -> str:
    """The text with everything that is not prose removed.

    The prompt figures reproduce, word for word, what the system sends to the model,
    and that text says ``cell``. Changing it would misreport the system, so the
    verbatim environments and inline code spans come out before the match rather than
    the word being allowed through.
    """
    text = path.read_text(encoding="utf-8")
    for env in ("tcolorbox", "verbatim", "lstlisting"):
        text = re.sub(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", " ", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", " ", text)          # markdown inline code
    text = re.sub(r"\\texttt\{[^}]*\}", " ", text)  # LaTeX inline code
    return text


def test_prose_avoids_retired_evaluation_vocabulary():
    paths = [ROOT / rel for rel in PROSE_FILES]
    for rel_root, glob in PROSE_GLOBS:
        paths += sorted((ROOT / rel_root).glob(glob))
    pattern = re.compile(r"\b(" + "|".join(RETIRED_PROSE) + r")\b", re.I)
    hits: list[str] = []
    for path in paths:
        for lineno, line in enumerate(_prose(path).splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()[:110]}")
    assert hits == []
