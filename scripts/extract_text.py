import pymupdf
from pathlib import Path


PDF_DIR = Path("text_data")
OUTPUT_DIR = Path("raw_text")


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = pymupdf.open(pdf_path)

    pages = []

    for page in doc:
        text = page.get_text()
        pages.append(text)

    doc.close()

    return "\n\n".join(pages)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.")

    successful = 0
    failed = 0

    for i, pdf_path in enumerate(pdf_files, start=1):

        output_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"

        try:
            text = extract_text_from_pdf(pdf_path)

            output_path.write_text(
                text,
                encoding="utf-8"
            )

            successful += 1

        except Exception as e:
            failed += 1
            print(f"\nERROR: {pdf_path.name}")
            print(e)

        if i % 100 == 0:
            print(f"Processed {i}/{len(pdf_files)}")

    print("\nExtraction completed.")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()