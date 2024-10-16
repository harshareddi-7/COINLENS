import requests
import streamlit as st
from streamlit_lottie import st_lottie


#Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="GetHeartSure", page_icon="🫀", layout="centered", initial_sidebar_state="collapsed")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style.css")



# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
with st.container():
    st.title(f":rocket: HEART DISEASE PREDICTION USING ML ALGORITHMS:rocket:")
    with st.expander("ABSTRACT"):
            st.write("For a long time, Cardiovascular diseases (CVD) is still one of the leading cause of death globally. The rise of new technologies such as Machine Learning (ML) algorithms can help with the early detection and prevention of developing CVDs. This study mainly focuses on the utilization of different ML models to determine the risk of a person in developing CVDs by using their personal lifestyle factors. This study used, extracted, and processed the 438,693 records as data from the Behavioral Risk Factor Surveillance System (BRFSS) in 2021 from World Health Organization (WHO). The data was then partitioned into training and testing data with a ratio of 0.8:0.2 to have an unknown data to evaluate the model that will be trained on. One problem that this study faced is the Imbalance among the classes and this was solved by using sampling techniques in order to balance the data for the ML model to process and understand well. The performance of the ML models was evaluated using 10-Stratified Fold cross-validation testing and the best model is Logistic Regression (LR) with F1 score of 0.32564. Logistic Regression model was then subjected to hyperparameter tuning and got the best score of 0.3257 with C = 0.1. Feature Importance was also generated from the LR model and the features that have the most impact is Sex, Diabetes, and the General Health of an individual. After getting the final LR model, it was then evaluated in the testing data and got a F1 score of 0.33. The Confusion Matrix was also used to better visualize the performance. And, the LR model correctly classified 79.18 % of people with CVDs and 73.46 % of people that is healthy. The AUC-ROC Curve was also used as a performance metric and the LR model got an AUC score of 0.837. The Logistic Regression model can be used in the medical field and can be utilized more by adding medical attributes to the data. Overall, this study gave us an insight and significant knowledge that can help in predicting the risk of CVDs by only using the personal attributes of an individual.")
            st.markdown("[READ FULL ARTICLE....](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))")
      
      #  st.
      

##with st.spinner("text"):

if st.button('Begin Prediction'):
    st.write('Medical Disclaimer:')
    st.write("The contents of this website are not intended to diagnose or treat any disease, or offer personal medical advice. You should seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read on this website.")
    st.write("The prediction provided by this model is for informational purposes only and should not be used to make any decisions regarding medical treatment or diagnosis. If you have a heart disease please consult your specified Doctor")
    
    
    if st.button('Agree'):
        
        st.page_link("page2.py")
    if st.button('Disagree'):
        st.stop()

# ---- CONTACT ----
with st.container():
    st.write("---")
    st.header("Get In Touch With Me!")
    st.write("##")

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    contact_form = """
    <form action="https://formsubmit.co/harshareddy95099@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()