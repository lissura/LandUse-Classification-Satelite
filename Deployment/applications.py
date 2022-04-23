import pickle
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Automatic Satelite Imagery Classificaiton",
    page_icon="🗺",
    layout="wide",
    initial_sidebar_state="expanded")
st.set_option('deprecation.showPyplotGlobalUse', False)

with open ("dictionary_class.pkl",'rb') as f:
    dict_ = pickle.load(f)

#swab key and value dictionary mapping
dict_ = {v:k for k, v in dict_.items()}
model = tf.keras.models.load_model("model_vgg.h5")
img = None

def load_prep(img):
    img_pred = Image.open(img).resize(size=(64,64))
    img_pred = np.asarray(img_pred)[None, :, :, :3]
    res = model.predict(img_pred)
    res = tf.keras.layers.Softmax()(res)

    return np.argmax(res, axis = 1)[0], res

st.markdown("""
<style>

res{background: #092235;
    border:1px solid;
    color: #888
    margin: 5px 0 0 0;
    font-size: 50px;
    line-height: 0.80;
    width: 100px;
    text-align: right;        
}


b{  background: #3F4F4F;
   padding: 25px 25px 25px 25px;
   border-radius: 10px;
   box-shadow: 0 0 20px 0 rgba(0, 0, 0, 0.2), 0 5px 5px 0 rgba(0, 0, 0, 0.24);
   color: #888
   margin: 5px 0 0 0;
   font-size: 70px;
   line-height: 0.80;
   width: 900px;
   text-align: center;
}

desc{
    font-size: 30px;
}
</style>

<b> 🗺 Land Use Classification Image 🗺 </b>
""", unsafe_allow_html=True)

st.write(" ")
st.write(" ")


st.markdown("""
<desc>
Please only upload a picture captured from satelite i.e Google Earth.
Further Explanation, visit my Github

[![](https://img.shields.io/badge/Land%20Classification%20-GitHub-10000?logo=github&logoColor=green)](https://github.com/lissura)

</desc>
""", unsafe_allow_html=True)


st.write("This website is still under development")

a, b = st.columns(2)
with a:
    file = st.file_uploader("", type=["jpg","png"])
with b:
    if file is not None:
        st.write("")
        st.image(file, use_column_width='auto')
        button = st.button("Classify")
        if button:
            # col4, col5 = st.columns([0.1, 1])
            # with col5:
                pred, probability = load_prep(file)
                st.write("\n")
                st.markdown(f""" <res> {dict_[pred]} Images <br>
                <br>
                        {int(probability[0, pred]*100):3d}% Confidence </res>""", unsafe_allow_html=True)
    else:
        buttons = st.button("Classify")
        if buttons :
            st.error("Please Upload First ❕❕❕")


# button= st.button("Predict")


# if file is not None:
#     button = st.button("Predict")
#     if button:
#             col1, col2, col3 = st.columns([0.5, 1, 0.1])
#             col4, col5 = st.columns([0.1, 1])
#             with col2:
#                     st.image(file, use_column_width='auto')
#                     pred, probability = load_prep(file)
#                     st.write("\n")
#             with col5:
#                     st.markdown(f""" <res> This is a {dict_[pred]} Images with 
#                         {int(probability[0, pred]*100):3d}% Confidence </res>""", unsafe_allow_html=True)