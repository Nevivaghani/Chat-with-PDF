import base64
from IPython.display import Image, display

def display_base64_image(base64_code):
    """Displays base64-encoded image in notebook (for dev use)."""
    image_data = base64.b64decode(base64_code)
    display(Image(data=image_data))

def get_images_base64(chunks):
    """Extract base64 images from chunks."""
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

def parse_docs(docs):
    from base64 import b64decode
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}
