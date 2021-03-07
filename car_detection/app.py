import streamlit as st
from utils import *

def main():
    st.title("Car Detection Computer Vision")
    test_file_path = "/opt/test-image.jpg"
    class_name_percentage = []
    bb_data = []
    predict_done = False

    uploaded_file = st.file_uploader("Choose a file")
    # To read file as bytes:
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        f = open(test_file_path, 'wb')
        f.write(bytes_data)
        f.close()
        with st.beta_container():
            st.image(test_file_path)

    col1, col2 = st.beta_columns(2)
    model_name = 'MobileNet'
    run_model = False
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

            bb_model = get_bb_model()
            bb_data = bb_model.predict(process_image(test_file_path))

            predict_done = True
        else:
            st.write("Working on ResNet")


    if predict_done:
        st.write("Prediction completed")
        st.write(class_name_percentage)

if __name__ == "__main__":
    main()
