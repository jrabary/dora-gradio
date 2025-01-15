import gradio as gr
import numpy as np
from dora import Node
from PIL import Image
import pyarrow as pa
import random


FPS = 1.0 / 5.0  # 5 FPS


def pyarrow_to_ndarray(struct_array: pa.StructArray) -> np.ndarray:
    """Convert a PyArrow struct array back to a numpy array.

    Args:
        struct_array: PyArrow struct array containing:
            - shape: List of dimensions
            - dtype: Original numpy dtype as string
            - data: Flattened array data

    Returns:
        Numpy array with original shape and dtype restored
    """
    shape = struct_array.field("shape")[0].as_py()
    dtype = np.dtype(struct_array.field("dtype")[0].as_py())
    data = np.array(struct_array.field("data")[0].as_py())
    return data.astype(dtype).reshape(shape)


def chatbot_response(instruction):
    # Placeholder function for chatbot response
    return f"Chatbot response to: {instruction}"


def main():
    im = Image.new("RGB", (640, 480), color="gray")
    im = np.array(im)

    node = Node()

    def handle_camera_streams():
        current_im = im
        while True:
            evt = node.next()
            if evt is not None:
                if evt["type"] == "INPUT":
                    if evt["id"] == "observations":
                        current_im = pyarrow_to_ndarray(evt["value"])
            else:
                break

            yield current_im

    def respond(message, chat_history):
        bot_message = random.choice(
            ["How are you?", "Today is a great day", "I'm very hungry"]
        )
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        message_to_send = pa.array([message])
        node.send_output("instruction", message_to_send)
        return "", chat_history

    with gr.Blocks() as interface:
        agent_view = gr.Image(label="Display Image")

        chatbot = gr.Chatbot(type="messages")
        instruction = gr.Textbox()
        # clear = gr.ClearButton([msg, chatbot])

        instruction.submit(respond, [instruction, chatbot], [instruction, chatbot])

        # instruction.submit(chatbot_response, inputs=instruction, outputs=chatbot)
        interface.load(handle_camera_streams, outputs=[agent_view], stream_every=FPS)

    interface.launch()


if __name__ == "__main__":
    main()
