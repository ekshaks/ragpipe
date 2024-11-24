from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, FormatOption
from docling_core.types.doc.labels import DocItemLabel

import json

'''
https://github.com/DS4SD/docling/issues/391#issuecomment-2492714238

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import FigureElement, InputFormat, Table
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling.datamodel.pipeline_options import PdfPipelineOptions
IMAGE_RESOLUTION_SCALE = 2.0

pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
'''

def pdf_to_md(pdf_path, out_file=None, redo=False):
    if not out_file:
        out_file = pdf_path.parent / f'{pdf_path.stem}.md'
    json_doc = pdf_path.parent / f'{pdf_path.stem}.json'
    
    if out_file.exists() and not redo:
        return
    
    headings: list[str] = [
            DocItemLabel.SECTION_HEADER,
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.TITLE,
    ]
    docling_options = {}
    converter = DocumentConverter(format_options=docling_options)
    result = converter.convert(pdf_path) #: ConversionResult
    md = result.document.export_to_markdown()
    with open(out_file, 'w') as fp: fp.write(md)

    with Path(json_doc).open("w") as fp:
        fp.write(json.dumps(result.document.export_to_dict(), 
                            indent=4,
                            )
                 ) 

def image_ids_to_md(images_dir, image_ids = [50, 78, 39], img_prefix='page_', img_fmt='png', out_dir=None):
   
    image_paths = [images_dir / f'{img_prefix}{id_}.{img_fmt}' for id_ in image_ids]
    if out_dir is None: out_dir = images_dir
    return image_files_to_md(image_paths, out_dir=out_dir)
 

def image_files_to_md(image_paths, out_dir=None):
    for img_path in image_paths:
        assert img_path.exists(), f'{img_path} does not exist.'
    docling_options = dict(
        images_scale = 1.0 #72 dpi
    )

    if out_dir is None: out_dir = image_paths[0].parent

    converter = DocumentConverter(format_options=docling_options)
    conv_results_iter = converter.convert_all(image_paths)
    md_files = []
    for i, doc in enumerate(conv_results_iter):
        md = doc.document.export_to_markdown()
        path = image_paths[i]
        out_file = out_dir / f'{path.stem}.md'
        with open(out_file, 'w') as fp: fp.write(md)
        md_files.append(out_file)
    return md_files

def image2md(image_reps, out_dir=None):
    image_files = [imrep.image_path for imrep in image_reps]
    return image_files_to_md(image_files, out_dir=out_dir)
