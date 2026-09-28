"""
clean_text.py - clean the raw myScheme text dumps.

Run from anywhere:
    python scripts/clean_text.py

Reads   <project>/raw_text/*.txt
Writes  <project>/cleaned_text/<scheme>.txt      (one file per UNIQUE scheme)
        <project>/cleaning_manifest.csv          (one row per raw file: status, title, warnings)

What is wrong with the raw files, and which step fixes it:
  1. Mojibake (â€œ, â‚¹, ï»¿ ...)        -> fix_encoding()   (ftfy + strip zero-width chars)
  2. Hard-wrapped lines (~130 chars)   -> unwrap()
  3. Website boilerplate at the top     -> parse_page()     (sign-out popup, nav bar, "Apply Now")
  4. Page repeated after the FAQ + footer -> parse_page()   (cut at "Sources And References")
  5. Section headings glued to text    -> parse_page()     ("BenefitsUnder the scheme")
  6. Duplicate files ("x.txt", "x copy.txt") -> main()     (hash of the cleaned text)
"""
import argparse
import csv
import hashlib
import re
from pathlib import Path

import ftfy  # pip install ftfy

ROOT = Path(__file__).resolve().parent.parent

SIGN_OUT = "Are you sure you want to sign out?"
NAV_SKIP = {"Sources And References", "Feedback"}          # nav items that are not content sections
END_PATTERN = re.compile(r"Sources And References|Was this helpful\?|News and Updates|©\s*20\d\d")
ZERO_WIDTH = dict.fromkeys(map(ord, "\ufeff\u200b\u200c\u200d\u2060\u00ad"), None)

# Only used by the fallback path, when a file does not follow the usual page layout.
BOILERPLATE = [
    r"Are you sure you want to sign out\?.*?Sign ?InBack",
    r"Something went wrong\. Please try again later\.Ok",
    r"You need to sign in before applying for schemes",
    r"It seems you have already initiated your application earlier\.To know more please visit",
    r"Cancel(?:Sign ?In|Apply Now|Sign Out)",
    r"Check Eligibility",
    r"©\s*20\d\d.*$",
]


# ---------- step 1 & 2: text-level cleaning ----------
def fix_encoding(text: str) -> str:
    text = ftfy.fix_text(text)          # repairs mojibake and turns curly quotes into plain quotes
    text = text.translate(ZERO_WIDTH)   # BOM, zero-width space, soft hyphen
    return text.replace("\xa0", " ")


def unwrap(text: str) -> str:
    # 'self-\nemployment' -> 'self-employment' (line was wrapped after a hyphen)
    text = re.sub(r"(?<=[A-Za-z])-\n(?=[A-Za-z])", "-", text)
    return re.sub(r"\s+", " ", text).strip()   # every other newline/tab/double space -> one space


# ---------- step 3, 4, 5: page structure ----------
def parse_page(text: str):
    """Return dict(title, state, tags, sections, warnings) or None if the layout is not recognised."""
    if SIGN_OUT not in text:
        return None
    warnings = []

    title = text.split(SIGN_OUT, 1)[0].strip()
    if not title:
        return None

    # Nav bar lists the sections this scheme has, e.g. DetailsBenefitsEligibilityExclusions...
    m = re.search(r"Sign ?InBack(.*?)Feedback", text)
    nav = re.split(r"(?<=[a-z])(?=[A-Z])", m.group(1)) if m else []
    sections = [s.strip() for s in nav if s.strip() and s.strip() not in NAV_SKIP]
    if not sections:
        sections = ["Details"]
        warnings.append("nav_not_found")

    # The real page header starts at the 2nd occurrence of the title: <state/ministry><title><tags>Details
    start = text.find(title, len(title))
    if start == -1:
        return None
    state = re.split(r"Check Eligibility|Apply Now", text[:start])[-1].strip()
    after = text[start + len(title):]
    j = after.find("Details")
    if j == -1 or j > 200:
        tags_raw, body = "", after
        warnings.append("details_marker_not_found")
    else:
        tags_raw, body = after[:j], after[j + len("Details"):]
    tags = [t.strip() for t in re.split(r"(?<=[a-z])(?=[A-Z])", tags_raw) if t.strip()]

    # Drop everything from "Sources And References" on: repeated intro + footer.
    end = END_PATTERN.search(body)
    if end:
        body = body[:end.start()]
    else:
        warnings.append("end_marker_not_found")

    # Find each heading in order; headings are glued to the text before/after them.
    bounds, pos = [(sections[0], 0, 0)], 0        # (name, heading_start, content_start)
    for name in sections[1:]:
        i = body.find(name, pos)
        if i == -1:
            warnings.append(f"heading_not_found:{name}")
            continue
        bounds.append((name, i, i + len(name)))
        pos = i + len(name)

    result = {}
    for k, (name, _, content_start) in enumerate(bounds):
        content_end = bounds[k + 1][1] if k + 1 < len(bounds) else len(body)
        content = body[content_start:content_end].strip()
        if content:
            result[name] = content
    if not result:
        warnings.append("empty_body")

    return {"title": title, "state": state, "tags": tags, "sections": result, "warnings": warnings}


def fallback_page(text: str):
    """Layout not recognised: remove known boilerplate and keep the rest as one block."""
    for pattern in BOILERPLATE:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    title = text[:80]
    return {"title": title, "state": "", "tags": [], "sections": {"Details": text},
            "warnings": ["fallback_layout"]}


def render(page: dict) -> str:
    lines = [f"TITLE: {page['title']}",
             f"STATE_OR_MINISTRY: {page['state']}",
             f"TAGS: {', '.join(page['tags'])}",
             ""]
    for name, content in page["sections"].items():
        lines += [f"## {name}", content, ""]
    return "\n".join(lines).strip() + "\n"


# ---------- driver ----------
def canonical_stem(path: Path) -> str:
    """'25-ciss copy' -> '25-ciss'"""
    return re.sub(r"\s+copy(?:\s*\(?\d+\)?)?$", "", path.stem, flags=re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "raw_text")
    ap.add_argument("--out", type=Path, default=ROOT / "cleaned_text")
    ap.add_argument("--manifest", type=Path, default=ROOT / "cleaning_manifest.csv")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # originals first, so a "copy" is the one that gets skipped as duplicate
    files = sorted(args.raw.glob("*.txt"), key=lambda p: (" copy" in p.stem.lower(), p.name.lower()))
    seen_hash, used_names, rows = {}, set(), []
    raw_chars = clean_chars = 0

    for path in files:
        row = {"source_file": path.name, "out_file": "", "status": "", "title": "",
               "state_or_ministry": "", "n_sections": 0, "n_chars": 0, "duplicate_of": "", "warnings": ""}
        try:
            raw = path.read_text(encoding="utf-8-sig", errors="replace")
            raw_chars += len(raw)
            text = unwrap(fix_encoding(raw))
            page = parse_page(text)
            row["status"] = "ok"
            if page is None:
                page = fallback_page(text)
                row["status"] = "fallback"
            out_text = render(page)
            digest = hashlib.sha1(out_text.encode("utf-8")).hexdigest()

            row.update(title=page["title"], state_or_ministry=page["state"],
                       n_sections=len(page["sections"]), n_chars=len(out_text),
                       warnings=";".join(page["warnings"]))

            if digest in seen_hash:
                row["status"] = "duplicate"
                row["duplicate_of"] = seen_hash[digest]
            else:
                name, n = canonical_stem(path), 1
                while name in used_names:               # same name, different content
                    n += 1
                    name = f"{canonical_stem(path)}_v{n}"
                used_names.add(name)
                (args.out / f"{name}.txt").write_text(out_text, encoding="utf-8")
                seen_hash[digest] = f"{name}.txt"
                row["out_file"] = f"{name}.txt"
                clean_chars += len(out_text)
        except Exception as e:                           # never let one bad file stop 2800
            row["status"], row["warnings"] = "error", repr(e)
        rows.append(row)

    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    count = lambda s: sum(r["status"] == s for r in rows)
    flagged = sum(1 for r in rows if r["out_file"] and r["warnings"])
    print(f"raw files read      : {len(rows)}")
    print(f"unique schemes kept : {count('ok') + count('fallback')}")
    print(f"duplicates skipped  : {count('duplicate')}")
    print(f"fallback layout     : {count('fallback')}")
    print(f"errors              : {count('error')}")
    print(f"kept files w/ warnings: {flagged}  (see cleaning_manifest.csv)")
    if clean_chars:
        print(f"chars: {raw_chars:,} raw -> {clean_chars:,} cleaned (kept files only)")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()