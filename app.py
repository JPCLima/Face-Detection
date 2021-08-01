import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import io
import base64

# Import classifiers
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")


def get_image_download_link(img, filename):
    """Generates a link allowing the PIL image to be downloaded
    in:  PIL image
    out: href string
    """
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a style="color:#C33753;text-decoration: None;" href="data:file/jpg;base64,{img_str}" download="{filename}">Download result</a>'
    return href


def resize_photo(resize_size, image_file):
    """ Resize the photo and returns the original_img, resized_img and the array of the resized_img image"""
    original_img = Image.open(image_file)
    resized_img = original_img.resize(resize_size)
    new_img = np.array(resized_img.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    resized_array = Image.fromarray(img)
    return original_img, resized_img, resized_array


def get_photo_size(choice, image_file):
    if choice == 'A3 (4961 x 3508 px)':
        resize_size = (4961, 3508)
        return resize_photo(resize_size, image_file)

    elif choice == 'A4 (3508 x 2480 px)':
        resize_size = (3508, 2480)
        return resize_photo(resize_size, image_file)

    elif choice == 'A5 (2480 x 1748 px)':
        resize_size = (2480, 1748)
        return resize_photo(resize_size, image_file)

    elif choice == 'A6 (1748 x 1240 px)':
        resize_size = (2480, 1748)
        return resize_photo(resize_size, image_file)

    elif choice == 'A7 (1240 x 874 px)':
        resize_size = (1240, 874)
        return resize_photo(resize_size, image_file)

    elif choice == 'Business Card (1004 x 650 px)':
        resize_size = (1004, 650)
        return resize_photo(resize_size, image_file)

    return resize_size


@st.cache
def load_image(img):
    return Image.open(img)


def detector(img_input, detetion_type):
    new_img = np.array(img_input.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if detetion_type == 'faces':
        detections = face_cascade.detectMultiScale(gray_img, 1.1, 2)
    elif detetion_type == 'eyes':
        detections = eye_cascade.detectMultiScale(gray_img, 1.1, 2)

    # Draw rectangle
    for (x, y, w, h) in detections:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 15)
    return img, detections


def main():

    st.title("Image Manipulation")

    # Main options
    options = ['Detections', 'Image Style', 'Resize', 'About']
    choice = st.sidebar.selectbox("Select Option", options)

    if choice == 'Image Style':
        st.subheader('Detections')

        # Upload file
        image_file = st.file_uploader(
            'Upload Image',  type=['jpg', 'png', 'jpeg'])

        enhance_type = st.sidebar.radio("Enhance Type",
                                        ["Original", "Gray-Scale", "Contrast", "Brightness"])

        if image_file is not None:
            original_img = Image.open(image_file)
            new_img = np.array(original_img.convert('RGB'))
            img = cv2.cvtColor(new_img, 1)
            st.text("Original Image")
            st.image(original_img)

            if enhance_type == 'Gray-Scale':
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                st.image(gray_img)

            if enhance_type == 'Contrast':
                c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
                enhancer = ImageEnhance.Contrast(original_img)
                contrast_img = enhancer.enhance(c_rate)
                st.image(contrast_img)

            if enhance_type == 'Brightness':
                b_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
                enhancer = ImageEnhance.Brightness(original_img)
                contrast_img = enhancer.enhance(b_rate)
                st.image(contrast_img)

    elif choice == 'Detections':
        st.subheader('Detection')

        options = ["Faces", "Eyes"]
        choice = st.sidebar.selectbox("Select Detection", options)

        image_file = st.file_uploader(
            'Upload Image',  type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            original_img = Image.open(image_file)
            new_img = np.array(original_img.convert('RGB'))
            img = cv2.cvtColor(new_img, 1)
            st.text("Original Image")
            st.image(original_img)

            if st.button("Process"):
                if choice == 'Faces':
                    result_img, result_faces = detector(original_img, 'faces')
                    st.image(result_img)
                    st.success(f'Found {len(result_faces)} faces')

                elif choice == 'Eyes':
                    result_img, result_eyes = detector(original_img, 'eyes')
                    st.image(result_img)
                    st.success(f'Found {len(result_eyes)} eyes')

    elif choice == 'Resize':
        st.subheader('Resize')

        options = ['A3 (4961 x 3508 px)', 'A4 (3508 x 2480 px)',
                   'A5 (2480 x 1748 px)', 'A6 (1748 x 1240 px)',
                   'A7 (1240 x 874 px)', 'Business Card (1004 x 650 px)']
        choice = st.sidebar.selectbox("Select Detection", options)

        # Upload file
        image_file = st.file_uploader(
            'Upload Image',  type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            original_img, resized_img, result = get_photo_size(
                choice, image_file)

        st.text("Original Image")
        st.image(original_img)
        st.image(resized_img)

        #st.text(f'Image Size: {original_img.size}')
        #st.text(f'Resized Image Size: {resized_img.size}')

        st.markdown(get_image_download_link(
            result, image_file.name), unsafe_allow_html=True)

    elif choice == 'About':
        st.subheader('About')


if __name__ == '__main__':
    main()
