import streamlit as st
import json
import os
from typing import Dict

import boto3
from langchain_community.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint

SAMPLE_MODEL_PARAMETERS = {
    "max_new_tokens": 1024,
    "top_p": 0.9,
    "temperature": 0.6,
    "stop": "<|eot_id|>"
}


def create_sm_client():
    # 从环境变量读取AWS凭证
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

    # 创建 boto3 会话
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name='us-east-1'  # 替换为您要使用的 AWS 区域
    )

    # 创建 SageMaker 客户端
    sagemaker_client = session.client('sagemaker-runtime', region_name="us-east-1")

    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
            input_str = json.dumps({"inputs": prompt, "parameters": model_kwargs})
            return input_str.encode("utf-8")

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            return response_json["generated_text"]

    content_handler = ContentHandler()
    llm = SagemakerEndpoint(
        endpoint_name="jumpstart-dft-meta-textgeneration-llama-3-8b-instruct",
        client=sagemaker_client,
        # streaming=True,
        model_kwargs=SAMPLE_MODEL_PARAMETERS,
        content_handler=content_handler,
    )

    return llm


with st.sidebar:
   """
   sample promptes:
   1. what is the recipe of mayonnaise?
   2. 给我计算菲波纳吉数列的python code
   """

st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

client = create_sm_client()
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    input = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    st.chat_message("user").write(prompt)
    print("call it " + prompt)
    # response = client.invoke(prompt)
    # print(response)
    msg = client.invoke(prompt)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
