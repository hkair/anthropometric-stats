# import libraries
import fitz
import io
import os
from PIL import Image
  
# STEP 2
# file path you want to extract images from
file = "../data/ansur/Gordon_2012_ANSURII_a611869.pdf"
output_directory = "../images/pdf_img"
  
# open the file
pdf_file = fitz.open(file)
  
def get_pixmaps_in_pdf(pdf_filename, output):
    pdf_file = fitz.open(pdf_filename)
    
    for page_index in range(len(pdf_file)):
        # get the page itself
        page = pdf_file[page_index]
        image_list = page.get_images(full=True)
        
        if page_index >= 54 and page_index <= 324:
            # printing number of images found in this page
            if image_list:
                print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
            else:
                # if no image is found, just continue
                print("[!] No images found on page", page_index)
                continue
            
            line_tokenize = page.get_text().split('\n')
            # ' ', '45 ', ' ', '(1) ABDOMINAL EXTENSION DEPTH, SITTING ' - 3rd index
            # ' ', '47 ', ' ', '(2) ACROMIAL HEIGHT ' - 3rd index
            print(line_tokenize)
            
            if line_tokenize[3] == ' ':
                name = "".join([ ch for ch in line_tokenize[4][4:].lower() if ch.isalnum()])
            else:
                name = "".join([ ch for ch in line_tokenize[3][4:].lower() if ch.isalnum()])
            print(name)
            for image_index, img in enumerate(image_list, start=1):
                bbox = page.get_image_bbox(img)
                pmap = page.get_pixmap(dpi=200, clip=bbox)
                pmap.save(os.path.join(output, f"{name}_{image_index}.png"))

    pdf_file.close()

get_pixmaps_in_pdf(file, output_directory)