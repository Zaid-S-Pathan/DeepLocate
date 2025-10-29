import streamlit as st
import requests
from streamlit_lottie import st_lottie
import cv2
from simple_facerec import SimpleFacerec
from apicall import add_in_base
from imagesapi import getimages


getimages()
# lottie adder
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()




# st.set_page_config(layout="wide")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)





# sidebar

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Face Recognition App to Find Missing People')
st.sidebar.subheader('Parameters')


app_mode = st.sidebar.selectbox('Choose the App mode',
['Detection Mode','Further Process']
)

if app_mode =='Further Process':
    st.markdown("<h1 style='font-family:sans-serif; color:#282c34; font-size: 50px;font-weight:700'>Further Process</h1>",unsafe_allow_html=True)
    st.markdown('Here after detection from the web cam in the detection mode we can get the data related to the missing person with his current location and the time detected in the information section of the website ')
    st.markdown('If you want more information about the missing peoples please visit missing people on the website and click on tracked locations for getting information about missing persons that have been tracked or have been found somewhere through surveillance')
    st.image("assets//navbarscreenshot.PNG")
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    
else:
  use_webcam = st.sidebar.button('Use Webcam')
  original_title = '<p style="font-family:sans-serif; color:#282c34; font-size: 50px;text-align:center;font-weight:700">Face Recognition App</p>'
  st.markdown(original_title, unsafe_allow_html=True)



  lottie_url="https://assets10.lottiefiles.com/packages/lf20_2szpas4y.json"
  lottie_face=load_lottieurl(lottie_url)
  if use_webcam==False:
    st_lottie(lottie_face,key='face recognition')
  else:
    # encode faces from a folder
    FRAME_WINDOW=st.image([])

    sfr=SimpleFacerec()
    sfr.load_encoding_images("images/")


    # load camera
    cap=cv2.VideoCapture(0)


    namesset=set()

    while True:
        ret,frame=cap.read()
        if not ret or frame is None:
            st.warning("⚠️ Could not read frame from webcam.") # changes done at og
            continue

        # detect faces
        face_locations,face_names=sfr.detect_known_faces(frame)
        for face_loc,name in zip(face_locations,face_names):
            y1,x2,y2,x1=face_loc[0],face_loc[1],face_loc[2],face_loc[3]
            if(name!="Unknown"):
                flag=name in namesset
                namesset.add(name)
                if(flag==False):
                    add_in_base(name)
                    print(name)


            cv2.putText(frame,name,(x1,y1-10),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,200),2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,200),4)


        cv2.imshow("Frame",frame)
        newframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(newframe)   
        key=cv2.waitKey(1)
        if key==27:
            break


    cap.release()
    cv2.destroyAllWindows()



# import streamlit as st
# import requests
# from streamlit_lottie import st_lottie
# import cv2
# from simple_facerec import SimpleFacerec
# from apicall import add_in_base
# from imagesapi import getimages


# # Function to load Lottie animation
# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()


# # Hide Streamlit elements (menu, header, footer)
# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)

# # Sidebar configuration
# st.sidebar.title('Face Recognition App to Find Missing People')
# st.sidebar.subheader('Parameters')

# app_mode = st.sidebar.selectbox('Choose the App mode',
#                                 ['Detection Mode', 'Further Process'])

# if app_mode == 'Further Process':
#     st.markdown("<h1 style='font-family:sans-serif; color:#282c34; font-size: 50px;font-weight:700'>Further Process</h1>", unsafe_allow_html=True)
#     st.markdown('Information on detected missing persons will appear here.')
# else:
#     use_webcam = st.sidebar.button('Use Webcam')
#     original_title = '<p style="font-family:sans-serif; color:#282c34; font-size: 50px;text-align:center;font-weight:700">Face Recognition App</p>'
#     st.markdown(original_title, unsafe_allow_html=True)

#     # Load Lottie animation if the webcam isn't used
#     lottie_url = "https://assets10.lottiefiles.com/packages/lf20_2szpas4y.json"
#     lottie_face = load_lottieurl(lottie_url)
#     if not use_webcam:
#         st_lottie(lottie_face, key='face recognition')
#     else:
#         # Start the webcam stream and encode faces from a folder
#         FRAME_WINDOW = st.image([])

#         sfr = SimpleFacerec()
#         sfr.load_encoding_images("images/")

#         # Load the camera
#         cap = cv2.VideoCapture(0)

#         # Set for names detected
#         namesset = set()

#         while True:
#             ret, frame = cap.read()

#             # Check if the frame is captured correctly
#             if not ret:
#                 st.error("Failed to capture image from the camera.")
#                 break

#             # Convert the frame from BGR (OpenCV default) to RGB (which face_recognition expects)
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             # Ensure the image is a valid 8-bit image
#             if rgb_frame.dtype != 'uint8':
#                 st.error("The image is not an 8-bit image. Converting to uint8.")
#                 rgb_frame = rgb_frame.astype('uint8')
            
#             # Handle different image depths (channels) to ensure it's either RGB or Grayscale
#             if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
#                 # RGB image, proceed
#                 pass
#             elif len(rgb_frame.shape) == 2:
#                 # Grayscale image, convert to RGB
#                 rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_GRAY2RGB)
#             else:
#                 st.error("Unsupported image type, must be 8-bit RGB or Grayscale.")
#                 continue

#             # Now pass it to the face_recognition function
#             try:
#                 face_locations, face_names = sfr.detect_known_faces(rgb_frame)
#             except RuntimeError as e:
#                 st.error(f"Error detecting faces: {e}")
#                 break

#             # Display detected faces and draw rectangles
#             for face_loc, name in zip(face_locations, face_names):
#                 y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
#                 if name != "Unknown":
#                     flag = name in namesset
#                     namesset.add(name)
#                     if not flag:
#                         add_in_base(name)
#                         print(name)

#                 cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

#             # Convert the frame back to RGB for Streamlit display
#             newframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             FRAME_WINDOW.image(newframe)

#             # Break loop if 'Esc' key is pressed
#             key = cv2.waitKey(1)
#             if key == 27:
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
