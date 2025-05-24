import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from googletrans import Translator

# Load the trained model
model = load_model('words-MobileNetV2.h5')

# Define class labels
class_labels = ['আজ', 'বাঘ', 'বাসা', 'বিয়োগ', 'বন্ধু', 'বৌদ্ধ', 'চামড়া', 'দাঁড়ানো', 'দাঁড়াও', 'দেশ', 
                'এখানে', 'গির্জা', 'বন্দুক', 'হকি', 'জেল', 'ক্যারাম', 'কিছুটা', 'কোথায়', 'অনুরোধ', 
                'পিয়ানো', 'পুরু', 'সাহায্য', 'সে', 'সমাজ কল্যাণ', 'সময়', 'সত্য', 'সুন্দর', 'স্যার', 
                'তারা', 'তুমি']

# Function to preprocess the frame for prediction
def preprocess_frame_for_prediction(frame):
    resized_frame = cv2.resize(frame, (128, 128))
    normalized_frame = resized_frame.astype(np.float32) / 255
    return np.expand_dims(normalized_frame, axis=0)

# Initialize a variable to store the sentence and to control prediction
sentence = []
is_predicting = False

# Function to make predictions and update the GUI
def make_prediction():
    global is_predicting
    if not is_predicting:
        return  # Stop prediction if the button is toggled off

    _, frame = cap.read()

    # Crop the top quarter right of the frame
    height, width, _ = frame.shape
    top_quarter_right = frame[:height//2, width//2:, :]

    preprocessed_frame = preprocess_frame_for_prediction(top_quarter_right)

    # Make prediction using the loaded model
    prediction = model.predict(preprocessed_frame)
    predicted_class = np.argmax(prediction)

    # Map the predicted class to a corresponding label
    predicted_label = class_labels[predicted_class]

    # Update the result text with the latest word prediction
    result_text.set(f"চিহ্নিত শব্দ: {predicted_label}")

# Function to add the current word to the sentence
def add_to_sentence():
    predicted_word = result_text.get().split(": ")[1]  # Get the predicted word
    sentence.append(predicted_word)
    sentence_text.set(f"বাক্য: {' '.join(sentence)}")

# Function to start/stop the prediction
def toggle_prediction():
    global is_predicting
    is_predicting = not is_predicting  # Toggle the prediction state
    if is_predicting:
        predict_button.config(text="বন্ধ করুন")  # Update button text to stop
    else:
        predict_button.config(text="চিহ্নিত করুন")  # Update button text to start

# Function to update the displayed image and check frame count for predictions
frame_count = 0  # Initialize a frame counter
def update_image():
    global frame_count

    _, frame = cap.read()

    # Convert the frame to RGB format for displaying with Tkinter
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    img = ImageTk.PhotoImage(image=img)

    panel.img = img
    panel.config(image=img)

    # Increment the frame count
    frame_count += 1

    # Make a prediction every 10 frames if prediction is active
    if is_predicting and frame_count % 10 == 0:
        make_prediction()

    # Continue updating the image in a loop
    root.after(10, update_image)

# Function to translate Bangla text to English
def translate_text(text):
    translator = Translator()
    translation = translator.translate(text, src='bn', dest='en')
    return translation.text

# Function to handle translation of the sentence
def translate_sentence():
    sentence_str = ' '.join(sentence)
    translated_text = translate_text(sentence_str)
    translation_result.set(f"ইংরেজি অনুবাদ: {translated_text}")

# Function to create the main interface
def create_main_interface():
    # Clear the current window
    for widget in root.winfo_children():
        widget.destroy()

    # Create a frame for the video feed on the left
    video_frame = tk.Frame(root)
    video_frame.grid(row=0, column=0, padx=10, pady=10, sticky='n')

    global panel
    panel = tk.Label(video_frame)
    panel.pack()

    # Create a frame for the controls on the right
    control_frame = tk.Frame(root, bg="#ffffff", bd=10)
    control_frame.grid(row=0, column=1, padx=20, pady=20, sticky='n')

    # Row weight for centered elements
    control_frame.grid_rowconfigure(0, weight=1)
    control_frame.grid_rowconfigure(1, weight=1)
    control_frame.grid_rowconfigure(2, weight=1)

    # Create a label to display the result (detected word)
    result_text.set("চিহ্নিত শব্দ: ")  # Default text
    result_label = tk.Label(control_frame, textvariable=result_text, font=font_large, bg="#ffffff", fg="#333333")
    result_label.grid(row=0, column=0, padx=10, pady=10)

    # Create a button to toggle prediction
    global predict_button
    predict_button = tk.Button(control_frame, text="চিহ্নিত করুন", command=toggle_prediction, **btn_style)
    predict_button.grid(row=1, column=0, padx=10, pady=10)

    # Create a label to display the sentence (accumulated words)
    sentence_text.set("বাক্য: ")  # Default text
    sentence_label = tk.Label(control_frame, textvariable=sentence_text, font=font_large, bg="#ffffff", fg="#333333", wraplength=400)
    sentence_label.grid(row=2, column=0, padx=10, pady=10)

    # Create a button to add the predicted word to the sentence
    add_word_button = tk.Button(control_frame, text="বাক্য যোগ করুন", command=add_to_sentence, **btn_style)
    add_word_button.grid(row=3, column=0, padx=10, pady=10)

    # Create a label to display the translation result
    translation_result.set("ইংরেজি অনুবাদ: ")  # Default text
    translation_label = tk.Label(control_frame, textvariable=translation_result, font=font_large, bg="#ffffff", fg="#333333")
    translation_label.grid(row=4, column=0, padx=10, pady=10)

    # Create a button to translate the accumulated sentence
    translate_button = tk.Button(control_frame, text="ইংরেজি", command=translate_sentence, **btn_style)
    translate_button.grid(row=5, column=0, padx=10, pady=10)

    # Run the update_image function to start capturing and displaying frames
    update_image()

# Home page function
def home_page():
    # Clear the current window
    for widget in root.winfo_children():
        widget.destroy()

    # Load and set background image for home page
    welcome_label = tk.Label(root, text="বাংলা সাংকেতিক ভাষা থেকে পাঠ্য এবং ইংরেজি অনুবাদক-এ আপনাকে স্বাগতম", font=('Helvetica', 24), bg="#f0f0f0", fg="black")
    welcome_label.pack(pady=20)

    start_button = tk.Button(root, text="শুরু করুন", command=create_main_interface, **btn_style)
    start_button.pack(pady=10)

# Initialize OpenCV capture object
cap = cv2.VideoCapture(0)

# Create the main window
root = tk.Tk()
root.title("Bangla Sign Language Recognition Interface")
root.geometry("800x600")
root.config(bg="#f0f0f0")  # Set background color

# Style configuration
font_large = ('Helvetica', 16)
font_medium = ('Helvetica', 14)
btn_style = {
    'font': font_medium,
    'bg': '#4CAF50',
    'fg': 'white',
    'activebackground': '#45a049',
    'relief': 'raised',
    'padx': 10,
    'pady': 5,
}

# Initialize text variables
result_text = tk.StringVar()
sentence_text = tk.StringVar()
translation_result = tk.StringVar()

# Display the home page
home_page()

# Run the Tkinter main loop
root.mainloop()

# Release the capture object when the program exits
cap.release()
