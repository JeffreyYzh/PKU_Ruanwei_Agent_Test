import os
import getpass
# 设置api key
os.environ["DASHSCOPE_API_KEY"] = "sk-baedb9f60d544dcfb37f6114e321a7c3"
from langchain.agents import AgentExecutor, Tool, initialize_agent, load_tools


from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI, Tongyi


llm = Tongyi()
llm.model_name = 'qwen-max'

class T2I:
    def __init__(self, device):
        import torch
        from diffusers import StableDiffusionXLPipeline
        print("Initializing T2I to %s" % device)
        # Load the model in half-precision
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "/mnt/bn/motor-cv-yzh/langchain/models/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b", torch_dtype=torch.float16
        )
        self.pipe = pipe.to("cuda")
        self.device = device

    def inference(self, prompt):
        image = self.pipe(prompt).images[0]
        save_dir = "/mnt/bn/motor-cv-yzh/langchain/sample/save_image.jpg"
        image.save(save_dir)
        print(f"Generated image has been saved to {save_dir}")

# 调用本地工具
t2i = T2I("cuda:0")
tools = load_tools(["ddg-search", "llm-math", "wikipedia"], llm=llm)
tools.append(Tool(name="Text to Image", func=t2i.inference, description="Generates image from text description."))
# 创建agent
agent = initialize_agent(tools,
                         llm,
                         agent="zero-shot-react-description",
                         verbose=True)
print(agent.agent.llm_chain.prompt.template)
query = """
Who is the current Chief AI Scientist at Meta AI? When was he born?
And generate a image drawing this person.
"""
agent.run(query)