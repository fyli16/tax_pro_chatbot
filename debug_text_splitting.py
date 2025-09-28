import fitz  # PyMuPDF
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_pymupdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def clean_pdf_text(text):
    # Replace multiple newlines with single spaces
    text = re.sub(r'\n+', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

# Extract text from PDF
pdf_text = extract_text_pymupdf("./i1040gi.pdf")

print("=== BEFORE CLEANING ===")
print("Text length:", len(pdf_text))
print("First 200 characters:", repr(pdf_text[:200]))
print("Sample text:")
print(pdf_text[:500])
print("\n" + "="*50 + "\n")

# Clean the text
pdf_text = clean_pdf_text(pdf_text)

print("=== AFTER CLEANING ===")
print("Text length:", len(pdf_text))
print("First 200 characters:", repr(pdf_text[:200]))
print("Sample text:")
print(pdf_text[:500])
print("\n" + "="*50 + "\n")

# Test text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.create_documents([pdf_text])

print("=== TEXT SPLITTING RESULTS ===")
print(f"Number of chunks: {len(chunks)}")
print(f"First chunk length: {len(chunks[0].page_content)}")
print("First chunk content:")
print(repr(chunks[0].page_content[:200]))
print("\nFirst chunk preview:")
print(chunks[0].page_content[:500])

# Check if chunks are reasonable
if len(chunks) > 0 and len(chunks[0].page_content) > 100:
    print("\n✅ Text splitting looks good!")
else:
    print("\n❌ Text splitting still has issues!")
