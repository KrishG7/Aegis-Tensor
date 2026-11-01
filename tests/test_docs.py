import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BANNER = (
    "> **Design history.** This document records an earlier plan. For current "
    "behaviour see the README, docs/THREAT_MATRIX.md and docs/DECISIONS.md."
)

HISTORY_DOCS = [
    "ARCHITECTURE.md",
    "TECH_STACK.md",
    "PHASES.md",
    "REQUIREMENTS.md",
    "TEAM_ALLOCATION.md",
    "ISSUE_7_ENGINEERING_DESIGN.md",
]

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"```.*?```", re.DOTALL)


def test_history_documents_carry_the_banner():
    for name in HISTORY_DOCS:
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        assert text.startswith(BANNER), f"docs/{name} does not start with the banner"


def test_readme_relative_links_exist():
    files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    missing = []
    for md in files:
        text = FENCE.sub("", md.read_text(encoding="utf-8"))
        for target in LINK.findall(text):
            target = target.strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path = target.split("#", 1)[0]
            if path and not (md.parent / path).exists():
                missing.append(f"{md.relative_to(ROOT)} -> {target}")
    assert not missing, "broken relative links:\n" + "\n".join(missing)
