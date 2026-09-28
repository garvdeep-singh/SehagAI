from pathlib import Path
import hashlib
import statistics

CLEANED_DIR = Path("cleaned_text")

MIN_CHARS = 200


def main():
    files = sorted(CLEANED_DIR.rglob("*.txt"))

    if not files:
        print("No cleaned files found.")
        return

    lengths = []
    empty_files = []
    short_files = []
    hashes = {}

    for file in files:
        text = file.read_text(
            encoding="utf-8",
            errors="replace"
        ).strip()

        length = len(text)
        lengths.append(length)

        if length == 0:
            empty_files.append(file.name)

        elif length < MIN_CHARS:
            short_files.append((file.name, length))

        digest = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        hashes.setdefault(digest, []).append(file.name)

    duplicate_groups = [
        names
        for names in hashes.values()
        if len(names) > 1
    ]

    print("=" * 60)
    print("CLEANED DATASET VALIDATION")
    print("=" * 60)

    print(f"Documents           : {len(files)}")
    print(f"Empty documents     : {len(empty_files)}")
    print(f"Documents < {MIN_CHARS} chars : {len(short_files)}")
    print(f"Duplicate groups    : {len(duplicate_groups)}")

    print()
    print(f"Minimum characters  : {min(lengths):,}")
    print(f"Maximum characters  : {max(lengths):,}")
    print(f"Average characters  : {statistics.mean(lengths):,.0f}")
    print(f"Median characters   : {statistics.median(lengths):,.0f}")

    print()

    if empty_files:
        print("EMPTY FILES:")
        for name in empty_files[:20]:
            print(f"  {name}")

    if short_files:
        print()
        print("SHORT FILES:")
        for name, length in short_files[:20]:
            print(f"  {name}: {length:,} chars")

    if duplicate_groups:
        print()
        print("DUPLICATE GROUPS:")
        for group in duplicate_groups[:10]:
            print("  " + ", ".join(group))

    print("=" * 60)


if __name__ == "__main__":
    main()