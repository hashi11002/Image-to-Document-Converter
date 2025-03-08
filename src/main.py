# pylint: skip-file

#imports
from PIL import Image  #For image handling
from pytesseract import pytesseract #OCR for text extraction
import numpy as np #Numpy for numerical operations
import cv2 #OpenCV for image processing
import argparse #for getting image path as a command line argument
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from docx import Document

  
# Defining paths to tesseract.exe 
path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Get screen resolution dynamically
screen_width = 1920  # Set manually or use an API
screen_height = 1080

'''
Preprocessing image to enhance OCR accuracy

Arguments: 
image_path (str): directory path to image file that needs to be processed.

Returns: 
PIL image obejct: Pillow image object ready for Optical Character Recognition(OCR)

'''
def preprocess_image(image_path):
    
    #Load image and convert it to greyscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 

    #Apply bilateral filtering to reduce noise while keeping edges sharp
    img = cv2.bilateralFilter(img, 5, 75,75)

    # Use Super-Resolution by resizing up (2x) and back down (to smooth edges)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(img) < 127:
        img = cv2.bitwise_not(img)
    
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 11, 2) # Convert to binary
    
    # Morphological Transformations (Dilation & Erosion to bolden text)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)  # Makes text thicker
    img = cv2.erode(img, kernel, iterations=1) 
    return Image.fromarray(img)  # Convert to PIL image

'''
Use tesseract to extract text from the image

Arguments: 
image_path (str): directory path to image file that needs to be processed.

Returns:
Text (str): Text that is tripped from the image using tesseract OCR
'''
def extract_text(image_path):
    """Extracts text from the given image."""
    img = preprocess_image(image_path)
    config = "--psm 6 --oem 3"
    text = pytesseract.image_to_string(img, lang="eng", config=config)
    return text.strip()

'''
Takes text and created a pdf with it

Arguments: 
string_to_pdf (str): Takes text that needs to be added to the pdf
output_pdf(str): output filename, default: output.pdf

Returns: 
Creates a pdf with the text passed in the folder path
'''
def create_pdf(string_to_pdf, output_path= "output.pdf"):
    pdf = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    lines = string_to_pdf.split("\n")
    y_position = height-50

    for line in lines:
        pdf.drawString(50, y_position, line)
        y_position -= 20
    
    pdf.save()
    

'''
Takes text and created a word document with it

Arguments: 
string_to_word (str): Takes text that needs to be added to the word doc
output_pdf(str): output filename, default: output.docx

Returns: 
Creates a word document with the text passed in the folder path
'''
def create_word(string_to_word, output_path= "output.docx"):
    doc = Document()

    doc.add_paragraph(string_to_word)
    doc.save(output_path)

'''
Main function
'''
def main():
    parser = argparse.ArgumentParser(description="Extract text from an image using Tesseract OCR.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("format", type=str, choices=["pdf", "word"], help="Select output format (pdf or docx).")
    parser.add_argument("--output", type=str, default="output", help="Output file name without extension.")

    args = parser.parse_args()
    extracted_text = extract_text(args.image_path)
    if args.format == "pdf":
        create_pdf(extracted_text, f"{args.output}.pdf")
    elif(args.format == "word" or args.format == "docx"):
        create_word(extracted_text, f"{args.output}.docx")
    else:
        print("Incorrect option. Use help for options")
    print("\nExtracted Text:\n", extracted_text)


if __name__ == "__main__":
    main()



