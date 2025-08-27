import os

def pdf2image():
    from pdf2image import convert_from_path
    filePath = '../highlighted_output.pdf'  # Adjusted for relative path
    images = convert_from_path(filePath, dpi=300)
    for i, image in enumerate(images):
        image.save(f'pdf2image_page_{i + 1}.png', 'PNG')

def pymupdf():
    import fitz  # PyMuPDF
    from PIL import Image

    filePath = '../highlighted_output.pdf'
    doc = fitz.open(filePath)
    zoom = 300 / 72  # Scale factor for 300 DPI
    matrix = fitz.Matrix(zoom, zoom)

    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=matrix)
        output_path = f'PyMuPDF_page_{i + 1}.png'
        pix.save(output_path)
        with Image.open(output_path) as im:
            im.save(output_path, dpi=(300, 300))

def pdfium(): 
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument("../highlighted_output.pdf")
    for i in range(len(pdf)):
        page = pdf[i]
        image = page.render(scale=4).to_pil()
        image.save(f"pdfium_image_{i + 1:03d}.png", format="PNG", dpi=(300, 300))

def wand(): 
    from wand.image import Image
    from wand.color import Color
    pdf_path = '../highlighted_output.pdf'
    output_prefix = "wand_image"
    resolution = 300

    with Image(filename=pdf_path, resolution=resolution) as pdf:
        for i, page in enumerate(pdf.sequence):
            with Image(page) as img:
                img.background_color = Color("white")
                img.alpha_channel = 'remove'
                img.format = 'png'
                img.compression_quality = 95
                img.save(filename=f"{output_prefix}-{i + 1}.png")

# Run all conversions
pdf2image()
pymupdf()
pdfium()
wand()
# The above code converts a PDF file into images using four different libraries: pdf2image, PyMuPDF, pypdfium2, and Wand.