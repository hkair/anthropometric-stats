# import libraries
import fitz
import io
import os
import re
from PIL import Image
  
# STEP 2
# file path you want to extract images from
file = "../data/ansur/Gordon_2012_ANSURII_a611869.pdf"
  
# open the file
pdf_file = fitz.open(file)
output_file = "../description.txt"

def split_on_empty_lines(s):
    # greedily match 2 or more new-lines
    blank_line_regex = r"(?:\r?\n){2,}"

    return re.split(blank_line_regex, s.strip())
  
def getVariableDescription(pdf_filename):
    
    pdf_file = fitz.open(pdf_filename)
    descriptions = []
    
    for page_index in range(len(pdf_file)):
        # get the page itself
        page = pdf_file[page_index]
        image_list = page.get_images(full=True)
        
        if page_index >= 54 and page_index <= 323:
            # printing number of images found in this page
            if image_list:
                print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
            else:
                # if no image is found, just continue
                print("[!] No images found on page", page_index)
                continue
            
            paragraph_tokenize = page.get_text().split("\n \n")
            sentence = paragraph_tokenize[1]
            title = sentence.split("\n")[0]
            if title.isspace():
                title = " ".join(sentence.split())
              
            name = "".join([ c for c in "".join([ token.lower() for token in title.split(" ") if not any(ch.isdigit() for ch in token)]) if c.isalnum()])
            description = paragraph_tokenize[1] + "\n" + paragraph_tokenize[2]
            descriptions.append(name)
            descriptions.append("\n")
            descriptions.append(description+"\n")
            descriptions.append("\n")
            
    with open(output_file , 'w', encoding='utf-8') as f:
        for s in descriptions:
            f.write(s)
        
    pdf_file.close()
    
getVariableDescription(file)