import streamlit as st
from utils import *

def show_annotation(uploaded_file, test_file_path):
    df = get_class_bb_maping(uploaded_file.name)
    st.pyplot(show_image_with_bb(df, test_file_path))

def show_prediction(class_name_percentage, bb_data, test_file_path):
    st.write("Predicted Class")

    st.pyplot(show_predicted_image_with_bb(test_file_path, bb_data))

    col1, col2 = st.beta_columns(2)
    col1.subheader("Predicted Class Name")
    col2.subheader("Prediction Percentage")
    for i in class_name_percentage:
        col1.write(i[0])
        col2.progress(i[1].item())

def main():
    st.title("Car Detection Computer Vision")
    test_file_path = "/opt/test-image.jpg"
    class_name_percentage = []
    bb_data = []
    run_model = False

    uploaded_file = st.file_uploader("Choose a file")
    # To read file as bytes:
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        f = open(test_file_path, 'wb')
        f.write(bytes_data)
        f.close()
        with st.beta_container():
            st.image(test_file_path)
        if st.button("Annotate"):
            show_annotation(uploaded_file, test_file_path)

    col1, col2 = st.beta_columns(2)
    model_name = 'MobileNet'
    with col1:
        model_name = st.radio(
            "Which Model do you want to try out?",
            ('MobileNet', 'ResNet')
        )
    with col2:
        if st.button("Classify"):
            st.write('I am running the model here')
            st.write(model_name)
            run_model = True

    if run_model:
        if model_name == "MobileNet":
            classify_model = get_classify_mobile()
            predict_data = classify_model.predict(process_image(test_file_path))
            class_name_percentage = retrieve_class_name(predict_data)

            bb_model = get_bb_mobile()
            bb_data = bb_model.predict(process_image(test_file_path))

            show_annotation(uploaded_file, test_file_path)
            show_prediction(class_name_percentage, bb_data, test_file_path)
        else:
            st.write("Working on ResNet")


if __name__ == "__main__":
    main()
