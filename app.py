from qa_on_pic import *

import gradio as gr

def demo_func(input_img):
    img_path = "used.png"
    Image.fromarray(input_img).save(img_path)
    qa_df = generate_qa(img_path, ppstructure_bin_path, table_engine)
    return {"output": qa_df.values.tolist()}

example_sample = []
for img_p in os.listdir("imgs/"):
    if img_p.endswith(".jpg"):
        img_p = os.path.join("imgs/" ,img_p)
        assert os.path.exists(img_p)
        example_sample.append(img_p)


demo = gr.Interface(
        fn=demo_func,
        inputs="image",
        outputs="json",
        title=f"DocVQA dataset generate 🍩 demonstration",
        examples=example_sample if example_sample else None,
    )

demo.launch(server_name=None, server_port=None)
