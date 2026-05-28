import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.callbacks.manager import CallbackManager
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler


# ------------------------------
# Environment + Logging
# ------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
ENDPOINT = os.getenv("endpoint")
api_key=os.getenv("api_key")
opc_project_id=os.getenv("opc_project_id")
COMPARTMENT_ID = os.getenv("COMPARTMENT_ID")


# ------------------------------
# OCI Generative AI model
# ------------------------------
grok_model = ChatOpenAI(
    model="xai.grok-4-1-fast-reasoning",
    base_url=ENDPOINT,
    api_key=api_key,
    default_headers={
        "opc-project-id": opc_project_id,
    },
    temperature=0.5,
    max_tokens=16000,
)

# ------------------------------
# Langfuse callback handler
# ------------------------------
langfuse_handler = CallbackHandler()
callback_manager = CallbackManager([langfuse_handler])

# ------------------------------
# Helpers
# ------------------------------
async def summarize_tool_output(text: str, model):
    if not text or len(text) < 1000:
        return text

    logging.info("Summarizing long tool output...")
    chain = load_summarize_chain(model, chain_type="stuff")
    try:
        docs = [Document(page_content=text[:8000])]
        summary = await chain.arun(docs)
        return "Summary of tool output:\n" + summary
    except Exception as e:
        logging.warning(f"Summarization failed: {e}")
        return text[:700] + "\n...[truncated output]..."


# ------------------------------
# FastAPI app
# ------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)


# ------------------------------
# Request model
# ------------------------------
class ChatInput(BaseModel):
    text: str


# ------------------------------
# FULL SYSTEM PROMPT (UNCHANGED)
# ------------------------------
SYSTEM_PROMPT = ("""
    You are a precise OCI DevOps assistant. Always use the provided tools to answer questions about Oracle Cloud Infrastructure (OCI).
    Always use structured tool calls, never print them as text.
    Always call tools in sequential order, wait for tool results before proceeding.
    Always invoke suitable tools for any answer.
    If information is unavailable through tools, respond with "'I do not have that information available through OCI APIs.'
    Never invent compartments, instances, or configurations.
    Your job is to understand user requests and automatically select and call the correct tools to complete the task.
    Always check for tool response; if the tool response contains 'error' or indicates a failure, analyze and retry with corrected parameters.
    Continue reasoning until all steps are complete.
    Provide the final answer in plain English with relevant results and send results in email notification. "
    For remediating scanned vulnerabilities you have to create bash script with sudo and apply mapped ELSA using tool remediate_vulnerabilities with only bash script.
    Sample bash script is :
    #!/bin/bash
    sudo yum update --advisory ELSA-XXXX -y && sudo reboot
""")


# ------------------------------
# Chat endpoint
# ------------------------------
@app.post("/chat")
async def chat_endpoint(input: ChatInput):
    text = input.text

    try:
        logging.info(f"Opening MCP session to {MCP_SERVER_URL}")

        async with streamablehttp_client(url=MCP_SERVER_URL) as (reader, writer, _):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                logging.info("MCP session initialized")

                # Load MCP tools
                tools = await load_mcp_tools(session)
                logging.info(f"Loaded {len(tools)} MCP tools")

                # Create agent (NO callbacks here for LangGraph)
                agent = create_react_agent(grok_model, tools)

                config = {
                    "configurable": {"thread_id": "default"},
                    "callbacks": callback_manager,
                }

                logging.info("Invoking LangGraph agent")

                response = await agent.ainvoke(
                    {
                        "messages": [
                            SystemMessage(content=SYSTEM_PROMPT),
                            HumanMessage(content=text),
                        ]
                    },
                    config=config,
                )

                messages = response.get("messages", []) if isinstance(response, dict) else response

                ai_msgs = [m for m in messages if m.__class__.__name__ == "AIMessage"]
                tool_msgs = [m for m in messages if m.__class__.__name__ == "ToolMessage"]

                if ai_msgs and getattr(ai_msgs[-1], "content", "").strip():
                    return {"reply": ai_msgs[-1].content}

                elif tool_msgs:
                    tool_output = getattr(tool_msgs[-1], "content", "")
                    final_reply = await summarize_tool_output(tool_output, grok_model)
                    return {"reply": final_reply}

                return {"reply": "Sorry, I did not get a valid response."}

    except Exception as e:
        logging.exception("Chat endpoint failure")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {}, 200

@app.get("/ready")
async def readiness_check():
    try:
        # Basic readiness check: MCP server URL reachable (optional)
        if not MCP_SERVER_URL:
            return {}, 503

        # You can add more checks here if needed
        return {}, 200
    except Exception:
        return {}, 503
