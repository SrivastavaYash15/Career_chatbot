import os
import fitz  #  import for PyMuPDF

base_dir = "C:/Users/yashs/OneDrive/Desktop/Career_Chatbot/pdfs"  # Folder with PDFs

# Open the output file ONCE outside the loop to keep appending content from all PDFs
with open("output.txt", "w", encoding="utf-8") as out:
    for filename in os.listdir(base_dir):
        if filename.endswith(".pdf"):
            fullpath = os.path.join(base_dir, filename)  # Full path to the PDF
            doc = fitz.open(fullpath)  # Open the PDF
            for page in doc:
                text = page.get_text()  # This gives you a string — no need to encode
                out.write(text)  # Write the string directly
                out.write("\n\n--- End of Page ---\n\n")  # Optional: separator between pages
