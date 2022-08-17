import numpy as np
import gradio as gr
import cv2 

def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189], 
        [0.349, 0.686, 0.168], 
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img

# start gradio demo
with gr.Blocks() as demo:
    gr.Markdown("Detect objects in a given image using Yolo")
    with gr.Row():
        image_input = gr.Image()
        image_output = gr.Image()
    image_button = gr.Button("Detect objects")
    image_button.click(sepia, inputs=image_input, outputs=image_output)

demo.launch()
