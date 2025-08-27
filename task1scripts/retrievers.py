



# PDF Processing
def pdfPlumber():
    import pdfplumber
    all_words = []
    with pdfplumber.open("highlighted_output.pdf") as pdf:
        for page in pdf.pages:
            text = page.extract_text() 
            all_words.extend(text.split())

def pypdf2():
    from PyPDF2 import PdfReader
    # PDF Processing
    words3 = []
    reader = PdfReader("highlighted_output.pdf")
    for page in reader.pages:
        text = page.extract_text() 
        words3.extend(text.split())

def pymupdf():
    import pymupdf
    words4 = [] # initialize an empty list to store words
    doc = pymupdf.open("highlighted_output.pdf") # open a document
    for page in doc: # iterate the document pages
        text = page.get_text() # get plain text encoded as UTF-8
        words4.extend(text.split()) # extend the list without reassigning

def pdfMiner():
    words5 = []
    from pdfminer.high_level import extract_text
    text = extract_text("highlighted_output.pdf")
    words5 = text.split()


def pdftotext():
    import pdftotext
    with open("highlighted_output.pdf", "rb") as f:
        pdf = pdftotext.PDF(f)
    words2 = []
    for page in pdf:
        words2.extend(page.split())

pdfPlumber()    
pypdf2()
pymupdf()
pdfMiner()
pdftotext()