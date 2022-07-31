import io
import requests
from PIL import Image   
import streamlit as st
import json
import numpy as np
from requests_toolbelt.multipart.encoder import MultipartEncoder


st.title("Web App for Segmentation, Detection, and Classification of Images")

backend_url = "http://fastapi:8000/"

prediction_type = st.sidebar.selectbox('Single or Batch Prediction', ('Single', 'Batch'))

model_choose = st.sidebar.selectbox('Select Model', ('Segmentation', 'Classification', 'Detection'))



def process(image, server_url: str):

    r = requests.post(
        server_url, files={"image": ("filename", image, "image/jpeg")}
    )

    return r
backend = "http://fastapi:8000/segmentation"

if (prediction_type == 'Single'):

    input_image = st.file_uploader("Insert Image Here:") 
    if st.button("Get Result!"):
    

        if input_image:
        
            if(model_choose == 'Segmentation'):
                col1, col2 = st.columns(2)
                segments = process(input_image, backend)
                original_image = Image.open(input_image).convert("RGB")
                segmented_image = Image.open(io.BytesIO(segments.content)).convert("RGB")
                col1.header("Original")
                col1.image(original_image, use_column_width=True)
                col2.header("Segmented")
                col2.image(segmented_image, use_column_width=True)

            elif(model_choose == 'Classification'):
                col1, col2 = st.columns(2)
                #classify = requests.post('http://fastapi:8000/classification', files=input_image, timeout=8000)
                original_image = Image.open(input_image).convert("RGB")
                #image = Image.open(io.BytesIO(classify.content)).convert("RGB")
                #data=json.dumps(classify.json())
                #predictions = data
                #col1.header(predictions)
                #col1.image(original_image, use_column_width=True)

                files = {"file": input_image.getvalue()}
                res = requests.post(f"http://fastapi:8000/classification", files=files)
                prediction = res.json()
                #image = Image.open(img_path.get("name"))
                prediction = json.dumps(prediction)
                prediction = prediction.split('"')[3]
                col1.header(prediction)
                col1.image(original_image, use_column_width=True)



            elif(model_choose == 'Detection'):
                detect = process(input_image, backend_url + 'detection')
                detected_image = Image.open(io.BytesIO(detect.content)).convert("RGB")
                st.header("Output with Confidence Score:")
                st.image(detected_image, use_column_width=True)
                #st.write("Evaluation Metric (Confidence):", labels)

        else:
            st.write("Insert an image!")


#elif(prediction_type == 'Batch'):
#    if(model_choose == 'Segmentation'):
#        st.write("Dataset is")
#        st.progress()



#    elif(model_choose == 'Classification'):
#        st.write("Dataset is")

                
#    elif(model_choose == 'Detection'):
#       st.write("Dataset is")
