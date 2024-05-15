from pdf2image import convert_from_path
from PIL import Image
from datetime import datetime
import os
import cv2
import numpy as np
import pandas as pd

# Path to the folder containing PDF files
pdf_images_folder = '/Users/alaaabdelazeem/Desktop/Projects/AI/DentalXRayAgeEstimate-CNN/data/raw'
output_folder = '/Users/alaaabdelazeem/Desktop/Projects/AI/DentalXRayAgeEstimate-CNN/data/preprocessed'
df = pd.DataFrame(columns=["image", "age"])
i=0
for pdf_file in os.listdir(pdf_images_folder):
    if pdf_file.endswith('.pdf'):
        #Check if duplicate skip
        ending= pdf_file.split('.')[0][-3:] 
        if(ending == "(2)"):
            continue

        pdf_path = os.path.join(pdf_images_folder, pdf_file)
       
        # Convert PDF to images
        image = convert_from_path(pdf_path)
        #Save image
        image_path = os.path.join(output_folder, f"image{i}.png")
        image[0].save(image_path, 'PNG')
        #Extracting Date
        birth_date = pdf_file.split('_')[-1].split('.')[0][-10:] 
        try:
            birth_date_obj = datetime.strptime(birth_date, '%Y-%m-%d')
            age = (datetime.now() - birth_date_obj).days / 365.25
            df.loc[i] = [f"image{i}.png", age]
        except:
            print(f"Error Format in File {pdf_file}")
        i+=1

df.to_csv('/data/image_ages.csv', index=False)