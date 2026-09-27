import os

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    Form,
    Request,
    WebSocket,
    WebSocketDisconnect
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from typing import Annotated


# -------------------------
# Environment configuration
# -------------------------

load_dotenv()

api_key = os.getenv(
    "OPENAI_API_SECRET_KEY"
)

if not api_key:
    raise RuntimeError(
        "OPENAI_API_SECRET_KEY is not configured."
    )

client = OpenAI(
    api_key=api_key
)


# -------------------------
# FastAPI configuration
# -------------------------

app = FastAPI(
    title="Full-Stack AI Assistant"
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(
    directory="templates"
)


# -------------------------
# Chat configuration
# -------------------------

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Provide clear, accurate, and concise answers to user questions. "
    "Help users with programming, software development, technology, "
    "learning, problem-solving, and general questions."
)

chat_log = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]

chat_responses = []


# -------------------------
# WebSocket chat
# -------------------------

@app.websocket("/ws")
async def websocket_chat(
    websocket: WebSocket
):
    await websocket.accept()

    try:

        while True:

            user_input = (
                await websocket.receive_text()
            ).strip()

            if not user_input:
                continue


            # Store user message
            chat_log.append({
                "role": "user",
                "content": user_input
            })

            chat_responses.append(
                f"You: {user_input}"
            )


            # Stream AI response
            ai_response = ""

            stream = client.responses.create(
                model="gpt-5.6",
                input=chat_log,
                stream=True
            )

            for event in stream:

                if (
                    event.type
                    == "response.output_text.delta"
                ):

                    delta = event.delta

                    ai_response += delta

                    await websocket.send_text(
                        delta
                    )


            # Store completed AI response
            chat_log.append({
                "role": "assistant",
                "content": ai_response
            })

            chat_responses.append(
                f"AI Assistant: {ai_response}"
            )


    except WebSocketDisconnect:

        print(
            "WebSocket client disconnected."
        )

    except Exception as e:

        print(
            f"WebSocket error: {e}"
        )

        try:
            await websocket.send_text(
                "Sorry, something went wrong "
                "while generating the response."
            )
        except Exception:
            pass


# -------------------------
# Chat page
# -------------------------

@app.get(
    "/",
    response_class=HTMLResponse
)
async def chat_page(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={
            "chat_responses":
                chat_responses
        }
    )


# -------------------------
# Traditional POST chat
# -------------------------

@app.post(
    "/",
    response_class=HTMLResponse
)
async def chat(
    request: Request,
    user_input: Annotated[str, Form()]
):

    user_input = user_input.strip()

    if not user_input:

        return templates.TemplateResponse(
            request=request,
            name="home.html",
            context={
                "chat_responses":
                    chat_responses
            }
        )


    chat_log.append({
        "role": "user",
        "content": user_input
    })

    chat_responses.append(
        f"You: {user_input}"
    )


    response = client.responses.create(
        model="gpt-5.6",
        input=chat_log
    )

    bot_response = response.output_text


    chat_log.append({
        "role": "assistant",
        "content": bot_response
    })

    chat_responses.append(
        f"AI Assistant: {bot_response}"
    )


    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={
            "chat_responses":
                chat_responses
        }
    )


# -------------------------
# Image page
# -------------------------

@app.get(
    "/image",
    response_class=HTMLResponse
)
async def image_page(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="image.html",
        context={
            "image_url": None
        }
    )


# -------------------------
# Generate image
# -------------------------

@app.post(
    "/image",
    response_class=HTMLResponse
)
async def create_image(
    request: Request,
    user_input: Annotated[str, Form()]
):

    user_input = user_input.strip()

    image_url = None
    error_message = None

    try:

        response = client.responses.create(
            model="gpt-5.6",
            input=user_input,
            tools=[
                {
                    "type":
                        "image_generation"
                }
            ]
        )


        image_base64 = None

        for output in response.output:

            if (
                output.type
                == "image_generation_call"
            ):

                image_base64 = (
                    output.result
                )

                break


        if image_base64:

            image_url = (
                "data:image/png;base64,"
                + image_base64
            )

        else:

            error_message = (
                "No image was generated. "
                "Please try another prompt."
            )


    except Exception as e:

        print(
            f"Image generation error: {e}"
        )

        error_message = (
            "Sorry, the image could not "
            "be generated. Please try again."
        )


    return templates.TemplateResponse(
        request=request,
        name="image.html",
        context={
            "image_url":
                image_url,

            "error_message":
                error_message,

            "prompt":
                user_input
        }
    )