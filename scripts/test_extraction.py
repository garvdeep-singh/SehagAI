import pymupdf
from pathlib import Path


PDF_DIR = Path("text_data")

# Get the first PDF
pdf_files = list(PDF_DIR.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found.")
    exit()

pdf_path = pdf_files[0]

print(f"Testing: {pdf_path.name}")

doc = pymupdf.open(pdf_path)

print(f"Pages: {len(doc)}")

for page_number, page in enumerate(doc, start=1):
    text = page.get_text()

    print(f"\n--- PAGE {page_number} ---\n")
    print(text[:3000])

doc.close()