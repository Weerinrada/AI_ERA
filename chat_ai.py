import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.chat_models import BedrockChat
from langchain.tools import DuckDuckGoSearchRun
import boto3
from botocore.config import Config

# Create a custom configuration with increased timeout
custom_config = Config(
    read_timeout=400,  # 5 minutes
    connect_timeout=400,  # 5 minutes
    retries={"max_attempts": 3},
)

SYSTEM_PROMPT = """คุณเป็นผู้ช่วยที่เก่งทางด้านการวิเคราะห์การเงิน การธนาคาร และเชี่ยวชาญด้านการทำ credit analysis สำหรับสินเชื่อรถ (new car, used car, refinance 2W, refinance 4W, new motorbike, new big bike)
คุณมีความรู้ลึกซึ้งเกี่ยวกับการประเมินความเสี่ยงทางการเงิน การวิเคราะห์งบการเงิน และกระบวนการอนุมัติสินเชื่อรถยนต์ 
ให้คำแนะนำที่เป็นมืออาชีพและข้อมูลที่ถูกต้องแม่นยำเกี่ยวกับหัวข้อเหล่านี้ โดยให้คำตอบเป็นภาษาไทย"""

with st.sidebar:
    AWS_ACCESS_KEY_ID = st.text_input("AWS Access Key", type="password")
    AWS_SECRET_ACCESS_KEY = st.text_input("AWS Secret Key", type="password")
    AWS_SESSION_TOKEN = st.text_input("AWS Session Token", type="password")

    aws_region = st.text_input("AWS Region", value="ap-southeast-1")
    "[Get AWS credentials](https://d-966771b2b1.awsapps.com/start/#/?tab=accounts)"
    # "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/2_Chat_with_search.py)"
    # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("🔎 ผู้ช่วยวิเคราะห์การเงินและสินเชื่อรถ")

"""
ยินดีต้อนรับสู่ผู้ช่วยวิเคราะห์การเงินและสินเชื่อรถอัจฉริยะ! พร้อมตอบคำถามและให้คำแนะนำเกี่ยวกับการวิเคราะห์การเงิน การทำ credit analysis และสินเชื่อรถ
"""
# """
# In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
# Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
# """

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "สวัสดีค่ะ เราเป็นผู้ช่วยที่เชี่ยวชาญด้านการวิเคราะห์การเงิน การธนาคาร และการทำ credit analysis สำหรับสินเชื่อรถ มีอะไรให้เราช่วยเหลือไหมคะ?",
        }
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(
    placeholder="ถามคำถามเกี่ยวกับการวิเคราะห์การเงิน สินเชื่อรถ หรือ credit analysis ได้เลย"
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        st.info("Please add your AWS credentials to continue.")
        st.stop()

    bedrock_runtime = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        # region_name="us-east-1",
        region_name="ap-southeast-1",
        config=custom_config,
    )
    llm = BedrockChat(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        client=bedrock_runtime,
        model_kwargs={"system": SYSTEM_PROMPT},
    )
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent(
        [search],
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
