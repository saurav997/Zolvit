import PyPDF2 as pypdf2
from PIL import Image

def pdf_to_jpg(pdf_path, output_path):
    """Converts a PDF file to a JPG image using pypdf2.

    Args:
        pdf_path (str): The path to the PDF file.
        output_path (str): The path to save the JPG image.
    """

    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = pypdf2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)

        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]

            # pypdf2 doesn't have a direct 'get_image' method
            resources = page.get_resources()

            # Extract and handle images based on resource type
            if '/XObject' in resources:
                xobjects = resources.get('/XObject')
                for xobject_name, xobject in xobjects.items():
                    if xobject.get('/Subtype') == '/Image':
                        # Extract image data and create PIL Image
                        image_data = xobject.getData()
                        image_mode = xobject.get('/Filter')  # May need conversion
                        image_size = (xobject.get('/Width'), xobject.get('/Height'))  # Check for keys

                        pil_image = Image.frombytes(image_mode, image_size, image_data)
                        output_filename = f"{output_path}_page{page_num + 1}.jpg"
                        pil_image.save(output_filename, format="JPEG")
                        print(f"Page {page_num + 1} converted to {output_filename}")
            else:
                print(f"No image found on page {page_num + 1}")

# Example usage (assuming pypdf2 is installed)
pdf_path = "Jan to Mar\INV-136_Rishabh Ramola.pdf"
output_path = "output_image"
pdf_to_jpg(pdf_path, output_path)