
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from torchvision import transforms
import pandas as pd
from PIL import Image
import model #File containing the model
import image_preprocessor as ip #File containing the preprocessing functions
import plotly.express as px

st.title("Scribbled Numbers Classifier and Translator")
st.write(This application takes in the input of the number
# Run in GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Input the model path here
model = model.initialize_model(r'MNIST_Digit_Detector.pt')

st.subheader("Scribble any number on the canvas, and the corresponding number is given in Devanagari script, ")

# Create a canvas component
image_data = st_canvas(stroke_width=10,
                       stroke_color='#000000',
                       background_color='#AAAAAA',
                       width=500,
                       height=200
                       )

# Take the image data attribute and store it in the same variable.
# PS. Thats the only thing we need
image_data = image_data.image_data


# Do something interesting with the image data
if image_data is not None:

    # Extract a list of digits
    digit_lis= ip.extract_digits(image_data)

    
    for snip in digit_lis:

        # Sometimes if we put a dot in the canvas or if the 
        # Canvas in empty on streamlit the NoneType Error is
        # generated. To check that this if condition is placed
        if snip is not None:
        
            pil_image = Image.fromarray(snip)       
            
            # Transform the image to fit the model inputs
            transform = transforms.Compose([
                transforms.Resize(size=(28,28)),
                transforms.ToTensor(),
            ])

            # transform the image
            transformed_image = transform(pil_image).to(device)

            # Unsqueeze the tensor to [1,1,28,28]
            transformed_image = torch.unsqueeze(transformed_image,dim=0)

            with torch.inference_mode():
                # Set the model to evaluation mode
                model.eval()
                
                # Predict the digit of the model
                y_logits = model(transformed_image)
                y_pred_prob = torch.softmax(y_logits,dim=1)
                
                conf = torch.max(y_pred_prob).item()*100
                digit = str(torch.argmax(y_pred_prob).item())

            # Invert the colors for proper display
            invert = ip.invert_colors_opencv(pil_image)

            # A Streamlit Container Widget
            with st.container():

                # 2 column
                col1,col2 = st.columns(2)

                # The first column will show the image and the highest predicted value
                with col1:
                    dict1={'0':'०', '1':'१', '2':'२', '3':'३', '4':'४', '5':'५', '6':'६', '7':'७', '8':'८', '9':'९'}
                    dict2={"0":"౦", "1":"౧", "2":"౨", "3":"౩", "4":"౪", "5":"౫", "6":"౬", "7":"౭", "8":"౮", "9":"౯"}
                    dict3={"0":"౦", "1":"௧", "2":"௨", "3":"௩", "4":"௪", "5":"௫", "6":"௬", "7":"௭", "8":"௮", "9":"௯"}
                    dict4={'0':'_', '1':'I', '2':'II', '3':'III', '4':'IV', '5':'V', '6':'VI', '7':'VII', '8':'VIII', '9':'IX'}
                    st.image(invert)

                # The second column will show a bar graph with the Confidence scores of each digit
                with col2:
                    st.write(f'The predicted Digit is {digit} with Confidence {conf:.2f}')
                    st.write(f'The predicted digit in देवनागरी = {dict1[str(digit)]}\nThe predicted digit in తెలుగు = {dict2[str(digit)]}\nThe predicted digit in தமிழ் = {dict3[str(digit)]}')
                  
                    y_pred_prob_numpy = y_pred_prob.squeeze().to('cpu').numpy()
              
                    df = pd.DataFrame(
                        y_pred_prob_numpy,
                        [0,1,2,3,4,5,6,7,8,9])
                    fig = px.pie(df)
                    st.plotly_chart(fig)


        
        
