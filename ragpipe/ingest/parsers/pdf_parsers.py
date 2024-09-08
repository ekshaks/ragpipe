from io import BytesIO
import requests

def download_pdf(url, out_file=None):
    response = requests.get(url)
    if response.status_code == 200:
        res = BytesIO(response.content)
        
        if out_file:
            with open(out_file, "wb") as f:
                f.write(res.read())
        else: return res
    else:
        raise Exception(f"Failed to download PDF: Status code {response.status_code}")
    
def parse_pdf_pypdf(file_path):
    try:
        import pypdf
    except ImportError:
        raise ImportError("pypdf is not installed. Please install it using pip install pypdf")
    import io

    from pypdf import PdfReader
    reader = PdfReader(file_path)
    page_texts = []
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        text = page.extract_text()
        page_texts.append(dict(text=text, page_num=page_number))
    return page_texts


def parse_pdf_mupdf(file_path):
    # check if mupdf is installed
    try:
        import mupdf
    except ImportError:
        raise ImportError("mupdf is not installed. Please install it using pip install PyMuPDF")
    import io

    # Open the PDF file
    with open(file_path, 'rb') as file: pdf_data = file.read()
    
    pdf_buffer = io.BytesIO(pdf_data)
    doc = mupdf.open(pdf_buffer)
    num_pages = mupdf.count_pages(doc)
    
    # Iterate through each page and extract text
    doc_array = []
    for page_num in range(num_pages):
        page = mupdf.load_page(doc, page_num)
        text = mupdf.get_text(page)
        doc_array.append(dict(text=text, page_num=page_num))

    # Close the PDF document
    mupdf.close(doc)
    
    return doc_array

async def pdf_to_markdown(file_path):
    import io
    # use marker to convert pdf to markdown
    try:
        from marker.convert import convert_single_pdf
        from marker.models import load_all_models
    except ImportError:
        raise ImportError("marker is not installed. Please install it using pip install marker")
    
    models = load_all_models()
    # read pdf file into BytesIO
    with open(file_path, 'rb') as file: pdf_data = file.read()
    pdf_buffer = io.BytesIO(pdf_data)
    # convert pdf to markdown
    markdown, _, _ = convert_single_pdf(pdf_buffer, models)
    yield markdown

def pdf_to_image(file_path, output_dir=None):
    print(f"Extracting images from {file_path} ==> {output_dir}")
    from pdf2image import convert_from_path
    images = convert_from_path(file_path)

    if output_dir is not None:
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
    
        for i, image in enumerate(images):
            image.save(f'{output_dir}/page_{i+1}.png', 'PNG')
            
    return images

'''
pip install unstructured_inference 
brew install poppler
'''
def pdf_to_section_tables_unstructured(file_path):
    #use unstructured to convert pdf to sections and tables

    import io
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        raise ImportError("unstructured is not installed. Please install it using pip install unstructured[pdfminer]")
    # read pdf file into BytesIO
    #with open(file_path, 'rb') as file: pdf_data = file.read()
    #pdf_buffer = io.BytesIO(pdf_data)
    # partition pdf into sections and tables
    sections = partition_pdf(file_path, strategy="hi_res")
    print(sections)
    return sections

def test():
    file_path = "./data/pitches/DDOG Investor Presentation Aug-24.pdf"
    from pathlib import Path
    assert Path(file_path).exists()
    #elements = pdf_to_section_tables(file_path); print(elements)
    #return elements
    pdf_to_image(file_path, Path("./data/pitches/images"))

if __name__ == "__main__":
    test()
