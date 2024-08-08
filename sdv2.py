# 导入必要的库
from xml.sax.xmlreader import InputSource
from langchain.agents import AgentExecutor, Tool, initialize_agent, load_tools
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI, Tongyi
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.tools import tool
from transformers import pipeline
from huggingface_hub import hf_hub_download
import torch
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import cv2
import os
import sys
import av
import torch
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
os.environ["DASHSCOPE_API_KEY"] = "sk-baedb9f60d544dcfb37f6114e321a7c3"
os.environ['http_proxy'] = "http://sys-proxy-rd-relay.byted.org:8118"
os.environ['https_proxy'] = "http://sys-proxy-rd-relay.byted.org:8118"
os.environ['no_proxy'] = "code.byted.org"
os.chdir("/mnt/bn/motor-cv-yzh/langchain")
sys.path.append("/mnt/bn/motor-cv-yzh/langchain")


# Step 1: 使用Hugging Face模型理解视频内容
class V2T:
    def __init__(self, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from diffusers import StableDiffusionPipeline
        from transformers import pipeline
        print("Initializing V2T to %s" % device)
        # Load the model in half-precision
        self.pipe = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", torch_dtype=torch.float16, device_map="auto", cache_dir="/mnt/bn/motor-cv-yzh/langchain/models")
        self.processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", cache_dir="/mnt/bn/motor-cv-yzh/langchain/models")
        self.pipe.to(device)
        self.device = device

    def read_video_pyav(self, container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def to_device(self, mydict):
        ans = dict()
        for x, y in mydict.items():
            ans[x] = y.to(self.device)
        return ans

    def inference(self, video_path):
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        video = self.read_video_pyav(container, indices)
        # For better results, we recommend to prompt the model in the following format
        prompt = "USER: <video>What is the video telling? ASSISTANT:"
        inputs = self.processor(text=prompt, videos=video, return_tensors="pt")
        inputs = self.to_device(inputs)
        out = self.pipe.generate(**inputs, max_new_tokens=60)
        output = self.processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].split("ASSISTANT: ")[-1][:-1]
        print(f"Processed V2T.run, video_path: {video_path}, text: {output}")
        return output

# Step 2: 使用Hugging Face模型生成图片
# @tool
class T2I:
    def __init__(self, device):
        import torch
        from diffusers import StableDiffusionXLPipeline
        print("Initializing T2I to %s" % device)
        # Load the model in half-precision
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, cache_dir="/mnt/bn/motor-cv-yzh/langchain/models"
        )
        self.pipe = pipe.to("cuda")
        self.device = device

    def inference(self, prompt):
        image = self.pipe(prompt).images[0]
        save_dir = "/mnt/bn/motor-cv-yzh/langchain/sample/save_image.jpg"
        image.save(save_dir)
        print(f"Generated image has been saved to {save_dir}")

class I2V:
    def __init__(self, device) -> None:
        import torch
        from diffusers import I2VGenXLPipeline
    
        pipeline = I2VGenXLPipeline.from_pretrained(
            "ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16", cache_dir="/mnt/bn/motor-cv-yzh/langchain/models"
        )
        pipeline.enable_model_cpu_offload()
        self.pipe = pipeline.to(device)

    def inference(self, prompt, image_dir):
        from diffusers.utils import export_to_gif, load_image
        image = load_image(image_dir).convert("RGB")
        prompt = prompt
        negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
        generator = torch.manual_seed(8888)

        frames = pipeline(
            prompt=prompt,
            image=image,
            num_inference_steps=50,
            negative_prompt=negative_prompt,
            guidance_scale=9.0,
            generator=generator,
        ).frames[0]
        save_dir = "/mnt/bn/motor-cv-yzh/langchain/sample/i2v.gif"
        video_path = export_to_gif(frames, save_dir)
        print(f"Generated image has been saved to {video_path}")


# 创建一个工具列表
class ConversationBot():
    
    def __init__(self) -> None:
        # 创建代理
        self.v2t = V2T("cuda:0")
        self.t2i = T2I("cuda:0")
        self.i2v = I2V("cuda:0")
        llm = Tongyi()
        llm.model_name = 'qwen-max'
        self.tools = [
            Tool(name="Video to Text", func=self.v2t.inference, description="Extracts text description from video."),
            Tool(name="Text to Image", func=self.t2i.inference, description="Generates image from text description."),
            Tool(name="Images to Video", func=self.i2v.inference, description="Generates video from images."),
        ]
        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.init_agent()

    def init_agent(self):
        prompt_template = PromptTemplate(
            input_variables=["video_path", "output_video_path"],
            template="""
            Given a video at {video_path}, first extract the text description from it. Then, generate an image based on the text description. 
            Finally, create a video from the generated image and save it at {output_video_path}.
            """
        )
        self.agent = initialize_agent(
                    llm=self.llm,
                    agent="conversational-react-description",
                    prompt_template=prompt_template,
                    memory=self.memory,
                    return_intermediate_steps=True,
                    tools=self.tools,
                    verbose=True
        )
            
    def __call__(self, query):
        res = self.agent(query)
        return res



# 初始化代理
if __name__ == "__main__":
    video_path = "/mnt/bn/motor-cv-yzh/langchain/sample/truck.mp4"
    agent = ConversationBot()
    print(agent.agent.agent.llm_chain.prompt.template)
    query = "The video path is /mnt/bn/motor-cv-yzh/langchain/sample/truck.mp4, please describe the video and generete an image according to the description of the video, finally you should genearate the video with the image and the description of the video."
    result = agent(query)
    print(f"Generated video saved at {result['output_video_path']}")