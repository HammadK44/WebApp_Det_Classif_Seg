import io
import requests
from PIL import Image   
import streamlit as st
import json
import numpy as np
from requests_toolbelt.multipart.encoder import MultipartEncoder
import random
from db import db




st.title("Web App for Segmentation, Detection, and Classification of Images")
model_choose = st.sidebar.selectbox('Select Model', ('Segmentation', 'Classification', 'Detection'))

if model_choose == 'Classification':
    prediction_type = st.sidebar.selectbox('Single or Batch Prediction', ('Single', 'Batch'))
else:
    prediction_type = 'Single'


#------------------------------------------------------------------------------------------------------



def process(image, server_url: str):

    r = requests.post(
        server_url, files={"image": ("filename", image, "image/jpeg")}
    )

    return r
segmentation_backend_url = "http://127.0.0.1:8000/segmentation"      # change "127.0.0.1" to "fastapi"
detection_backend_url = "http://127.0.0.1:8000/detection"            # when dockerizing



if (prediction_type == 'Single'):


    input_image = st.file_uploader("Insert Image Here:") 
    if st.button("Get Result!"):
        if input_image:
            UploadedFileName = input_image.name
        


            if(model_choose == 'Segmentation'):
                col1, col2 = st.columns(2)
                segments = process(input_image, segmentation_backend_url)
                original_image = Image.open(input_image).convert("RGB")
                segmented_image = Image.open(io.BytesIO(segments.content)).convert("RGB")
                col1.header("Original")
                col1.image(original_image, use_column_width=True)
                col2.header("Segmented")
                col2.image(segmented_image, use_column_width=True)

#------------------------------------------------------------------------------------------------------

            elif(model_choose == 'Classification'):
                x = db.collection_Single.find_one({"ImageName": UploadedFileName})
                original_image = Image.open(input_image).convert("RGB")
                y = str(x)

                if y == "None":
                    col1, col2 = st.columns(2)

                    files = {"file": input_image.getvalue()}
                    res = requests.post(f"http://127.0.0.1:8000/classification", files=files)
                    prediction = res.json()
                    prediction = json.dumps(prediction)
                    #prediction = prediction.split('"')[3]
                    number = random.randint(0,1000)
                    z = db.collection_Single.find_one({"PredictionNo": number})

                    if z == "None":
                        record_dict ={
                        "PredictionID": number,
                        "ImageName": UploadedFileName,
                        "Prediction": prediction
                        }                                                       # Stored in DB
                        insertInTable= db.collection_Single.insert_one(record_dict)

                    else:
                        number = random.randint(0,1000)
                        record_dict ={
                        "PredictionID": number,
                        "ImageName": UploadedFileName,
                        "Prediction": prediction
                        }                                                     
                        insertInTable= db.collection_Single.insert_one(record_dict)

                    col1.header("Prediction Retrieved from Model:")
                    col2.header(prediction)


                else:
                    col1, col2 = st.columns(2)
                    x["_id"] = str(x["_id"])
                    x = str(x)
                    x = x.split("'", 10)
                    x = x[-2]
                    prediction = '"' + x + '"'

                    col1.header("Prediction Retrieved from DB:")
                    col2.header(prediction)


                col1.image(original_image, use_column_width=True)


#------------------------------------------------------------------------------------------------------

            elif(model_choose == 'Detection'):
                
                detect = process(input_image, detection_backend_url)
                detected_image = Image.open(io.BytesIO(detect.content)).convert("RGB")
                st.header("Output with Confidence Score:")
                st.image(detected_image, use_column_width=True)
                #st.write("Evaluation Metric (Confidence):", labels)

        else:
            st.write("Insert an image!")



#------------------------------------------------------------------------------------------------------

elif(prediction_type == 'Batch'):


    if(model_choose == 'Classification'):
        if st.button("Validation Run!"):

            st.write("Dataset is of Skin Cancer Disease with 2 Classes")

            class_dataset = './dataset/test'
            getpath = 'YES'
            record_dict ={
                        "GetPath": getpath,
                        "DatasetPath": class_dataset
                        }  
            insertInTable= db.collection_Batch.insert_one(record_dict)

            #class_dataset = './dataset/test'
            #pathdataJSON = {
            #    'class_dataset':  class_dataset
            #}
            #response = requests.post(f"http://127.0.0.1:8000/batchclassification", json = pathdataJSON)

            #response = requests.post(f"http://127.0.0.1:8000/batchclassification")


            #metrics = response.json()
            #metrics = json.dumps(metrics)
            #st.write(response)


            x = db.collection_Batch.find_one({"Dataset": class_dataset})
            x["_id"] = str(x["_id"])
            x = str(x)
            x = x.split("'", 18)
            #x = x[-1]

            accuracy = x[10]
            accuracy = accuracy.split(",",2)[0]
            accuracy = accuracy.split(":",2)[1]

            loss = x[12]
            loss = loss.split(",",2)[0]
            loss = loss.split(":",2)[1]

            precision = x[14]
            precision = precision.split(",",2)[0]
            precision = precision.split(":",2)[1]

            recall = x[16]
            recall = recall.split(",",2)[0]
            recall = recall.split(":",2)[1]

            f1score = x[18]
            f1score = f1score.split("}",2)[0]
            f1score = f1score.split(":",2)[1]

            st.header("Batch Prediction Metrics:")

            st.write("Accuracy:")
            st.write(accuracy)

            st.write("Loss:")
            st.write(loss)

            st.write("Precision:")
            st.write(precision)

            st.write("Recall:")
            st.write(recall)

            st.write("F1Score:")
            st.write(f1score)




            


#    elif(model_choose == 'Classification'):
#        st.write("Dataset is")

                
#    elif(model_choose == 'Detection'):
#       st.write("Dataset is")
