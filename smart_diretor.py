# 导入必要的库
from statistics import mode
from xml.sax.xmlreader import InputSource
from langchain.agents import AgentExecutor, Tool, initialize_agent, load_tools
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI, Tongyi
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.tools import tool
from transformers import pipeline
from huggingface_hub import hf_hub_download
import torch
import diffusers
from diffusers.models import ControlNetModel
from diffusers.utils import load_image
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import cv2
import os
import sys
import av
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from sys_prompt import VISUAL_CHATGPT_PREFIX, VISUAL_CHATGPT_FORMAT_INSTRUCTIONS, VISUAL_CHATGPT_SUFFIX 
os.environ["DASHSCOPE_API_KEY"] = "sk-baedb9f60d544dcfb37f6114e321a7c3"
os.environ['http_proxy'] = "http://sys-proxy-rd-relay.byted.org:8118"
os.environ['https_proxy'] = "http://sys-proxy-rd-relay.byted.org:8118"
os.environ['no_proxy'] = "code.byted.org"
os.chdir("/mnt/bn/motor-cv-yzh/langchain")
os.environ["CUDA_VISIBLE_DEVICES"]="0"
sys.path.append("/mnt/bn/motor-cv-yzh/langchain")

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

# @prompts(name="Ask information about the image",
#              description="useful when you need an answer for a question based on an image. "
#                          "like: what is the background color of the last image, how many cats in this figure, what is in this figure. "
#                          "The input to this tool should be a comma separated string of two, representing the image_path and the question")
# class IVQA:
#     def __init__(self, device) -> None:
#         import torch
#         from lavis.models import load_model_and_preprocess
#         # loads BLIP-2 pre-trained model
#         self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", \
#             is_eval=True, device=device)
#         self.model.to(device)
#         self.device = device
        
#     def inference(self, inputs):
#         image_path, question = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
#         raw_image = Image.open(image_path).convert("RGB")
#         # prepare the image
#         image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
#         outputs = self.model.generate({"image": image, "prompt": f"Question: {question} Answer:"})
#         print(f"Processed I2T.run, image_path: {image_path}, text: {outputs}")
#         return outputs

@prompts(name="Ask information about the image",
             description="useful when you need an answer for a question based on an image. "
                         "like: what is the background color of the last image, how many cats in this figure, what is in this figure. "
                         "The input to this tool should be a comma separated string of two, representing the image_path and the question")
class IVQA:
    def __init__(self, device) -> None:
        import torch
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        self.processor = Blip2Processor.from_pretrained("/mnt/bn/motor-cv-yzh/cache/models--Salesforce--blip2-opt-2.7b/snapshots/235c75ea3861136b9dd202c6edc6a7ba285c35e3")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "/mnt/bn/motor-cv-yzh/cache/models--Salesforce--blip2-opt-2.7b/snapshots/235c75ea3861136b9dd202c6edc6a7ba285c35e3", \
            torch_dtype=torch.float16)
        self.model = model.to(device)

    def inference(self, inputs):
        image_path, question = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        raw_image = Image.open(image_path).convert("RGB")
        question = f"Question: {question} Answer:"
        inputs = self.processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)

        generated_ids = self.model.generate(**inputs)
        outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        print(f"Processed I2T.run, image_path: {image_path}, text: {outputs}")
        return outputs


@prompts(name="Answer Question About The Video",
             description="useful when you need an answer for a question based on a video. "
                         "like: what is the background color of the last image, how many cats in this figure, what is in this figure. "
                         "The input to this tool should be a comma separated string of two, representing the video_path and the question")
class VVQA:
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

    def inference(self, inputs):
        video_path, question = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        video = self.read_video_pyav(container, indices)

        # For better results, we recommend to prompt the model in the following format
        prompt = f"USER: <video>{question}? ASSISTANT:"
        inputs = self.processor(text=prompt, videos=video, return_tensors="pt")
        inputs = self.to_device(inputs)
        out = self.pipe.generate(**inputs, max_new_tokens=200)
        output = self.processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].split("ASSISTANT: ")[-1][:-1]
        print(f"Processed V2T.run, video_path: {video_path}, text: {output}")
        return output

@prompts(name="Generate Image From User Input Text",
             description="useful when you want to generate an image from a user input text and save it to a file. "
                         "like: generate an image of an object or something, or generate an image that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")
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
        return save_dir

@prompts(name="Generate Face Embedding and Keypoints",
             description="useful when you want to generate personalized image, you should firstly use the tool to distill face embedding and keypoints of a person."
                         "The input to this tool should be a string, representing the given image."
                         "The output is a comma separated string of two, representing the direction of face embedding and keypoints")
class FaceDetect:
    def __init__(self, device):
        from insightface.app import FaceAnalysis
        # prepare 'antelopev2' under ./models
        app = FaceAnalysis(name='antelopev2', root='/mnt/bn/motor-cv-yzh/langchain/', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        self.app = app

    def inference(self, inputs):
        from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
        # load an image
        
        face_image = load_image(inputs)

        # prepare face emb
        face_info = self.app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
        face_emb = face_info['embedding']   # 512,
        face_kps = draw_kps(face_image, face_info['kps'])

        emb_dir = "/mnt/bn/motor-cv-yzh/langchain/results/face_emb.npy"
        kps_dir = "/mnt/bn/motor-cv-yzh/langchain/results/kps.jpg"
        np.save(emb_dir, face_emb)
        face_kps.save(kps_dir)
        output = f"Face embedding has been saved to {emb_dir} and key points has been saved to {kps_dir}"
        print(output)
        return output


@prompts(name="Generate Personalized Image From Face Embedding, Keypoints and User Input Text",
             description="useful when you want to generate an pesonalized image from given face embedding, face keypoints and input text and save it to a file. "
                         "(but you must generate face embedding and keypoints before using the tool!)"
                         "like: generate an image including the indicated person following the instruction of texts."
                         "The input to this tool should be a comma separated string of three, representing the file direction of face embeddings and face keypoints, and the text description. The text description should be the form like [A photo of a person in PLACE].")
class PersonalizedT2I:
    def __init__(self, device):
        import diffusers
        from diffusers.models import ControlNetModel

        import cv2
        import torch
        import numpy as np
        from PIL import Image

        from insightface.app import FaceAnalysis
        from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

        # prepare models under ./checkpoints
        face_adapter = f'/mnt/bn/motor-cv-yzh/langchain/checkpoints/ip-adapter.bin'
        controlnet_path = "/mnt/bn/motor-cv-yzh/langchain/checkpoints/ControlNetModel"

        # load IdentityNet
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16, \
                    cache_dir="/mnt/bn/motor-cv-yzh/langchain/models")

        base_model = 'wangqixun/YamerMIX_v8'  # from https://civitai.com/models/84040?modelVersionId=196039
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            cache_dir="/mnt/bn/motor-cv-yzh/langchain/models"
        )
        pipe.to(device)
        # load adapter
        pipe.load_ip_adapter_instantid(face_adapter)
        self.pipe = pipe
        self.device = torch.device(device)
    
    def inference(self, inputs):
        face_emb_dir, face_kps_dir, prompt = inputs.split(",")[0], inputs.split(",")[1], ','.join(inputs.split(',')[2:][1:])
        
        face_emb = torch.tensor(np.load(face_emb_dir)).to(self.device)
        face_kps = load_image(face_kps_dir)

        # prompt
        prompt = prompt
        negative_prompt = "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vibrant, colorful, balck and white"

        # generate image
        image = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=0.5,
            ip_adapter_scale=0.3,
            num_inference_steps=30
        ).images[0]

        save_path = "/mnt/bn/motor-cv-yzh/langchain/results/pi.jpg"
        image.save(save_path)
        print(f"Generated image has been saved to {save_path}")
        return save_path


@prompts(name="Generate Video From User Input Image",
            description="useful when you want to generate a video with an image."
                        "like: make a static image alive."
                        "The input to this tool should be the image_path.")
class I2V:
    def __init__(self, device) -> None:
        import torch
        from diffusers import (
            DDIMScheduler,
            MotionAdapter,
            PIAPipeline,
        )
        adapter = MotionAdapter.from_pretrained("openmmlab/PIA-condition-adapter", cache_dir="/mnt/bn/motor-cv-yzh/langchain/models")
        pipe = PIAPipeline.from_pretrained("SG161222/Realistic_Vision_V6.0_B1_noVAE", motion_adapter=adapter, \
            cache_dir="/mnt/bn/motor-cv-yzh/langchain/models")
        # enable FreeInit
        # Refer to the enable_free_init documentation for a full list of configurable parameters
        pipe.enable_free_init(method="butterworth", use_fast_sampling=True)

        # Memory saving options
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe.to(device)
        self.device = device
    
    def inference(self, inputs):
        from diffusers.utils import export_to_gif, load_image
        image = load_image(inputs)
        image = image.resize((512, 512))
        prompt = "best quality, high quality"
        negative_prompt = "wrong white balance, dark, sketches, worst quality,low quality"

        generator = torch.Generator("cpu").manual_seed(0)

        output = self.pipe(image=image, prompt=prompt, generator=generator)
        frames = output.frames[0]
        save_dir = "/mnt/bn/motor-cv-yzh/langchain/results/i2v.gif"
        video_path = export_to_gif(frames, save_dir)
        output = f"Generated video has been saved to {video_path}"
        print(output)
        return output

# 创建一个工具列表
class ConversationBot():
    
    def __init__(self) -> None:
        # 创建代理
        self.v2t = VVQA("cuda:0")
        self.i2t = IVQA("cuda:0")
        self.t2i = T2I("cuda:0")
        self.i2v = I2V("cuda:0")
        self.fd = FaceDetect("cuda:0")
        self.pt2i = PersonalizedT2I("cuda:0")
        llm = Tongyi()
        llm.model_name = 'qwen-max'
        self.tools = [
            Tool(name=self.v2t.name, func=self.v2t.inference, description=self.v2t.description),
            Tool(name=self.i2t.name, func=self.i2t.inference, description=self.i2t.description),
            Tool(name=self.t2i.name, func=self.t2i.inference, description=self.t2i.description),
            Tool(name=self.i2v.name, func=self.i2v.inference, description=self.i2v.description),
            Tool(name=self.fd.name, func=self.fd.inference, description=self.fd.description),
            Tool(name=self.pt2i.name, func=self.pt2i.inference, description=self.pt2i.description),
        ]
        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.init_agent()

    def init_agent(self):
        PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = VISUAL_CHATGPT_PREFIX, VISUAL_CHATGPT_FORMAT_INSTRUCTIONS, VISUAL_CHATGPT_SUFFIX
        self.agent = initialize_agent(
                    llm=self.llm,
                    agent="conversational-react-description",
                    memory=self.memory,
                    return_intermediate_steps=True,
                    tools=self.tools,
                    verbose=True,
                    agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS,
                          'suffix': SUFFIX}, )
            
    def __call__(self, query):
        res = self.agent(query)
        return res



# 初始化代理
if __name__ == "__main__":
    agent = ConversationBot()
    print(agent.agent.agent.llm_chain.prompt.template)
    # query = "Where is the video [/mnt/bn/motor-cv-yzh/langchain/sample/demo_ref_video.mp4] taken place? " \
    #         "And generete a personalinzed image in which the subject comes from the image [/mnt/bn/motor-cv-yzh/InstantID/examples/musk_resize.jpeg] and is situated in the place mentioned above." \
    #         "Finally make the image live."

    query = "How is the animal in video [/mnt/bn/motor-cv-yzh/langchain/sample/tiger.mp4]" \
            "different from the animal in image [/mnt/bn/motor-cv-yzh/langchain/sample/cat.jpeg]."
    result = agent(query)