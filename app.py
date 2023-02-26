import streamlit as st
import torch
from PIL import Image
from io import *
import glob
from datetime import datetime
import os

## CFG for model weights
cfg_model_path = "best.pt"

## END OF CFG


def imageInput(src):
    
    if src == 'Upload your own data':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            #call Model prediction--
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=True) 
            model.cpu()
            pred = model(imgpath)
            pred.render()  # render box in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            #--Display predicton
            
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction', use_column_width='always')

    elif src == 'From test set': 
        # Image selector slider
        imgpath = glob.glob('data/images/*')
        imgsel = st.slider('Please select an image from the test set.', min_value=1, max_value=len(imgpath), step=1) 
        image_file = imgpath[imgsel-1]
        submit = st.button("Predict the selected image")
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:            
            if image_file is not None and submit:
                #call Model prediction--
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=True) 
                pred = model(image_file)
                pred.render()  # render box in image
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))
                #--Display predicton
                    img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))
                    st.image(img_, caption='Model Prediction')



def main():
    # -- Sidebar
    st.sidebar.title('⚙️Options:')
    st.sidebar.markdown("You may choose to use an image from my test dataset, or upload your own. Please select an input source:")
    datasrc = st.sidebar.radio(, ['From test set', 'Upload your own data'])
    
    # -- End of Sidebar

    st.header('🪪 ID Detection')
    st.subheader('Please choose an option on the left.')
    st.subheader('Legend for prediction labels: 0 - ID card, 1 - Passport')
    st.sidebar.markdown("You can find my streamlit deployment codes here: https://github.com/EdmondAng/streamlit-id-detection.git")
    imageInput(datasrc)

    

if __name__ == '__main__':
  
    main()