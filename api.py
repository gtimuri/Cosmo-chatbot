import os
import io
import json
import sqlite3
import random
import re
import tempfile
from uuid import uuid4
from datetime import datetime
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import asyncio
import base64
import wave

from openai import OpenAI

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

TTS_VOICES = {
    "en": "fable",
    "de": "fable",
    "ru": "fable"
}

_openai_client = None

def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required for TTS")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chromadb")
CHROMA_DB_COLLECTION_NAME = os.getenv("CHROMA_DB_COLLECTION_NAME", "knowledgebase")

CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", "./chat_history.db")

COSMO_FEEDBACK_PROMPT = """You are COSMO, a chatbot accompanying a scientific exhibition on robotics "Mensch, Roboter!" (Human, Robot!).
Your role is to act as a dialogical mediator between visitors and researchers.
Your main task is to invite visitors to share their experiences, opinions, expectations, and concerns related to the exhibition and its exhibits.
You acknowledge their contributions and ask open-ended follow-up questions to deepen their perspective.
You represent research as an open, evolving process that benefits from public input.
You are attentive, interested, and appreciative, but not authoritative.

CURRENT EXHIBITION PROJECT: {german_project_name} / {english_project_name}
CURRENT DIALOGUE STATE: {dialogue_state}
FEEDBACK TURN: {feedback_turn}/3

RELEVANT EXHIBITION CONTEXT (to help you ask informed questions about this project):
{context}

Interaction rules:
- Use an open, dialogical, and non-directive tone
- Encourage reflection through exploratory follow-up questions
- Do not evaluate responses as right or wrong
- Do not provide final conclusions or instructions
- Keep your answer to a maximum of about 300 characters (the part that answers the question).
- Answer in the same language as the user
- Do NOT use emojis

Turn-specific behavior (based on feedback_turn number):
- Turn 2: This is the visitor's FIRST response. Acknowledge their answer warmly and ask a follow-up question that deepens their perspective.
- Turn 3: This is the visitor's SECOND response. DO NOT ASK QUESTION! Thank them and use the EXACT closing phrase (translated to user's language): "Thank you for sharing this. Would you like to add anything else? Your perspective helps research."

If user input is completely unrelated to the exhibition or robotics, respond with:
"I can't answer that. I'm here to collect your thoughts and questions about the exhibition and pass them on to researchers."
"""

COSMO_QA_PROMPT = """You are COSMO, a chatbot accompanying a scientific exhibition on robotics "Mensch, Roboter!" (Human, Robot!).
A visitor has asked a question during the feedback collection. Answer their question, then return to collecting feedback.

CURRENT EXHIBITION PROJECT: {german_project_name} / {english_project_name}
CURRENT DIALOGUE STATE: {dialogue_state}
FEEDBACK TURN: {feedback_turn}/3

CRITICAL RULE: You MUST answer ONLY using the provided context below. Do NOT use your general knowledge about the topic. 
If the context says "No relevant information found" or doesn't contain information to answer the question, you MUST say that you don't have that information.

Interaction rules:
- Keep your answer to a maximum of about 300 characters (the part that answers the question).
- First, check the context below. If it says "No relevant information found for this project" or is empty, you MUST respond: "I don't have information about that in the exhibition materials for this project. Could you tell me what you found interesting about this exhibition instead?"
- If the context contains information: Answer briefly using ONLY the context. Do NOT add information from your general knowledge.
- Then, return to feedback collection with an appropriate question based on the current turn
- Answer in the same language as the user
- Do NOT use emojis

IMPORTANT: After answering, do NOT ask a question about your answer. Instead, ask a FEEDBACK question based on the turn:

Turn-specific endings (based on feedback_turn number):
- Turn 1: The user hasn't shared any feedback yet (they asked a question right at the start). After answering, ask an OPENING feedback question. Choose one from these and translate it to the user's language (try to not repeat the same question): {dialogue_starters_en}
- Turn 2: The user already shared one feedback response. After answering, ask a follow-up question to deepen their perspective about what they shared earlier.

EXHIBITION CONTEXT FOR THIS PROJECT:
{context}

Remember: Use ONLY this context. If the question cannot be answered from this context, say so and redirect to feedback collection.
"""

COSMO_COMPLETED_PROMPT = """You are COSMO, a chatbot accompanying a scientific exhibition on robotics "Mensch, Roboter!" (Human, Robot!).
The structured feedback collection is complete. You are now in free conversation mode.

CURRENT EXHIBITION PROJECT: {german_project_name} / {english_project_name}

Your behavior depends on what the visitor writes:

1. If they ask a QUESTION about the exhibition, robots, or technology:
   - Answer helpfully using the provided context
   - Keep your answer to a maximum of about 300 characters (the part that answers the question).
   - Context: {context}
   - In the end, ask a question to the user to continue the conversation. E.g: "Would you like to ask anything else?"

2. If they share additional FEEDBACK, opinions, or experiences:
   - Thank them briefly and warmly for sharing
   - Do not ask follow-up questions, but int the end ask a question to the user to continue the conversation. E.g: "Would you like to ask anything else?"
   - Example responses: "Thank you for sharing that perspective! Would you like to add something else or ask anything else?" or "I appreciate you telling me this. Your feedback will be passed on to the researchers. Would you like to add or ask anything else?"

Interaction rules:
- Answer in the same language as the user
- Do NOT use emojis
- Keep responses short and appreciative

If user input is completely unrelated to the exhibition or robotics, respond with:
"I can't answer that. I'm here to collect your thoughts and questions about the exhibition and pass them on to researchers."
"""

DIALOGUE_STARTERS = [
    "What stood out to you most in the exhibition?",
    "How did the exhibition make you think about robotics?",
    "Was there anything that surprised or unsettled you?",
    "What thoughts or questions are you leaving the exhibition with?",
    "What would you like researchers to know about your experience?",
]

DIALOGUE_STARTERS_DE = [
    "Was ist Ihnen in der Ausstellung am meisten aufgefallen?",
    "Wie hat die Ausstellung Ihr Denken über Robotik beeinflusst?",
    "Gab es etwas, das Sie überrascht oder verunsichert hat?",
    "Mit welchen Gedanken oder Fragen verlassen Sie die Ausstellung?",
    "Was möchten Sie den Forschern über Ihre Erfahrung mitteilen?",
]

DIALOGUE_STARTERS_RU = [
    "Что вам больше всего запомнилось на выставке?",
    "Как выставка повлияла на ваше представление о робототехнике?",
    "Было ли что-то, что вас удивило или обеспокоило?",
    "С какими мыслями или вопросами вы покидаете выставку?",
    "Что бы вы хотели, чтобы исследователи знали о вашем опыте?",
]

CLOSING_PROMPT = "Thank you for sharing this. Would you like to add anything else? Your perspective helps research."
CLOSING_PROMPT_DE = "Vielen Dank fürs Teilen. Möchten Sie noch etwas hinzufügen? Ihre Perspektive hilft der Forschung."
CLOSING_PROMPT_RU = "Спасибо, что поделились. Хотели бы вы добавить что-то еще? Ваша точка зрения помогает исследованиям."

COSMO_UNSURE_PROMPT = """You are COSMO, a chatbot accompanying a scientific exhibition on robotics "Mensch, Roboter!" (Human, Robot!).
The user's message was unclear or off-topic. You need to gently redirect them back to the conversation.

CURRENT EXHIBITION PROJECT: {german_project_name} / {english_project_name}
LAST BOT MESSAGE: "{last_bot_message}"

Your task:
- Politely acknowledge that you didn't fully understand what the user meant
- Gently ask a clarifying question to bring them back to the topic
- Keep your response short (2-3 sentences)
- Answer in the same language as the user
- Do NOT use emojis

Example responses:
- "I'm not sure I understood that correctly. Could you tell me more about what you meant in relation to the exhibition?"
- "Ich bin mir nicht sicher, ob ich das richtig verstanden habe. Könnten Sie mir mehr darüber erzählen, was Sie meinen?"
- "Не совсем понял(а), что вы имеете в виду. Могли бы вы уточнить, как это связано с выставкой?"
"""


class DialogueState:
    GREETING = "greeting"
    COLLECTING_FEEDBACK = "collecting_feedback"
    ANSWERING_QUESTION = "answering_question"
    COMPLETED = "completed"


def init_chat_db():
    conn = sqlite3.connect(CHAT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            chat_id TEXT PRIMARY KEY,
            german_project_name TEXT,
            english_project_name TEXT,
            created_at TEXT,
            dialogue_state TEXT DEFAULT 'greeting',
            feedback_turn INTEGER DEFAULT 0
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            FOREIGN KEY (chat_id) REFERENCES chats (chat_id)
        )
    ''')
    conn.commit()
    conn.close()


def get_db_connection():
    conn = sqlite3.connect(CHAT_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


_knowledgebase = None
_llm = None
_whisper_model = None


def get_knowledgebase():
    global _knowledgebase
    if _knowledgebase is None:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        _knowledgebase = Chroma(
            collection_name=CHROMA_DB_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH,
        )
    return _knowledgebase


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.7,
        )
    return _llm


def check_ffmpeg():
    import shutil
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg doesn`t found in your system.\n"
            "Please isntall ffmpeg:\n"
            "- Windows: choco install ffmpeg or download from https://ffmpeg.org/download.html\n"
            "- Linux: sudo apt install ffmpeg\n"
            "- macOS: brew install ffmpeg"
        )
    return ffmpeg_path


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model on device: {device}")
        _whisper_model = whisper.load_model("turbo", device=device)
    return _whisper_model


class CreateChatRequest(BaseModel):
    german_project_name: str
    english_project_name: str


class CreateChatResponse(BaseModel):
    chat_id: str
    german_project_name: str
    english_project_name: str
    created_at: str


class ChatMessageRequest(BaseModel):
    message: str


class StartDialogueRequest(BaseModel):
    language: str = "en"  # "en", "de", or "ru"


class GreetingResponse(BaseModel):
    message: str
    dialogue_state: str
    feedback_turn: int


class ChatInfo(BaseModel):
    chat_id: str
    german_project_name: str
    english_project_name: str
    created_at: str
    message_count: int
    dialogue_state: str
    feedback_turn: int


def get_chat_metadata(chat_id: str) -> Optional[dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM chats WHERE chat_id = ?', (chat_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def create_chat_record(chat_id: str, german_name: str, english_name: str, created_at: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO chats (chat_id, german_project_name, english_project_name, created_at, dialogue_state, feedback_turn) 
           VALUES (?, ?, ?, ?, ?, ?)''',
        (chat_id, german_name, english_name, created_at, DialogueState.GREETING, 0)
    )
    conn.commit()
    conn.close()


def get_dialogue_state(chat_id: str) -> tuple[str, int]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT dialogue_state, feedback_turn FROM chats WHERE chat_id = ?', (chat_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row["dialogue_state"] or DialogueState.GREETING, row["feedback_turn"] or 0
    return DialogueState.GREETING, 0


def update_dialogue_state(chat_id: str, state: str, feedback_turn: int = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if feedback_turn is not None:
        cursor.execute(
            'UPDATE chats SET dialogue_state = ?, feedback_turn = ? WHERE chat_id = ?',
            (state, feedback_turn, chat_id)
        )
    else:
        cursor.execute(
            'UPDATE chats SET dialogue_state = ? WHERE chat_id = ?',
            (state, chat_id)
        )
    conn.commit()
    conn.close()


def save_message(chat_id: str, role: str, content: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO messages (chat_id, role, content, timestamp) VALUES (?, ?, ?, ?)',
        (chat_id, role, content, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def get_chat_history(chat_id: str) -> list:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT role, content, timestamp FROM messages WHERE chat_id = ? ORDER BY id',
        (chat_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in rows]


def get_message_count(chat_id: str) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM messages WHERE chat_id = ?', (chat_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count


INTENT_CLASSIFICATION_PROMPT = """You are classifying a user message in a museum chatbot conversation.

CONTEXT:
- Current dialogue state: {dialogue_state}
- Current feedback turn: {feedback_turn}/3
- Last bot message: "{last_bot_message}"
- RAG context (exhibition info): "{rag_context_summary}"

USER MESSAGE: "{message}"

Classify into ONE of these categories:

1. "question" - The user is asking for EXPLANATION or wants to LEARN something. This includes:
   - Direct questions: "What is 6G?", "How does this robot work?"
   - Requests for explanation: "Explain the technology", "Tell me about..."
   - Expressing confusion and asking for help: "I don't understand why...", "I didn't understand...", "Я не понял, объясни", "Ich verstehe nicht..."
   Examples: "What is 6G?", "I didn't understand why AI is needed", "How does this work?", "Explain this to me"
   
2. "feedback" - The user is sharing their OPINION, EXPERIENCE, or IMPRESSION (not asking for information):
   - Sharing what they liked/disliked
   - Describing their feelings or reactions
   - Answering the bot's question about their experience
   Examples: "I found it fascinating", "The robots were cool", "I liked the soft robots", "It made me think about the future"

3. "unsure" - The user's message is:
   - Completely off-topic (not related to the exhibition at all)
   - Too vague to understand

IMPORTANT RULES:
- If the user says "I don't understand" or "I didn't understand" followed by a topic, this is a QUESTION (they want explanation), NOT feedback.
- "feedback" is only for opinions and impressions, NOT for requests for information or clarification.
- When user explicitly asks to explain something or says they don't understand, always classify as "question".

Respond with ONLY one word: "question", "feedback", or "unsure". Nothing else."""


def detect_intent_sync(
    message: str, 
    dialogue_state: str = "", 
    feedback_turn: int = 0,
    last_bot_message: str = "",
    rag_context: str = ""
) -> str:
    try:
        rag_summary = rag_context[:500] + "..." if len(rag_context) > 500 else rag_context
        
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=INTENT_CLASSIFICATION_PROMPT.format(
                dialogue_state=dialogue_state,
                feedback_turn=feedback_turn,
                last_bot_message=last_bot_message[:300] if last_bot_message else "None",
                rag_context_summary=rag_summary,
                message=message
            ))
        ])
        intent = response.content.strip().lower()
        
        if intent in ['question', 'feedback', 'unsure']:
            return intent
        
        return _detect_intent_fallback(message)
    except Exception as e:
        print(f"Intent detection error: {e}")
        return _detect_intent_fallback(message)


def _detect_intent_fallback(message: str) -> str:
    message_lower = message.lower().strip()
    
    question_patterns = [
        r'\?$',  # Ends with question mark
        r'^(what|how|why|when|where|who|which|can|could|is|are|do|does|will|would)\b',
        r'^(was|wie|warum|wann|wo|wer|welche|kann|können|ist|sind|hat|haben|wird|werden)\b',
        r'(tell me|explain|describe|erzähl|erklär|beschreib)',
        r'(i\s*don\'?t\s*understand|i\s*didn\'?t\s*understand|don\'?t\s*get\s*it|didn\'?t\s*get)',
        r'(ich verstehe nicht|versteh ich nicht|nicht verstanden)',
        r'(я не понял|не понимаю|объясни|расскажи|что это|как это работает)',
    ]
    
    for pattern in question_patterns:
        if re.search(pattern, message_lower):
            return 'question'
    
    return 'feedback'


async def detect_intent(
    message: str,
    dialogue_state: str = "",
    feedback_turn: int = 0,
    last_bot_message: str = "",
    rag_context: str = ""
) -> str:
    import asyncio
    return await asyncio.to_thread(
        detect_intent_sync, 
        message, 
        dialogue_state, 
        feedback_turn, 
        last_bot_message, 
        rag_context
    )


def retrieve_context(
    query: str, 
    german_project_name: str = "", 
    english_project_name: str = "", 
    k: int = 8
) -> str:
    try:
        kb = get_knowledgebase()

        filter_dict = None
        if english_project_name:
            filter_dict = {"title": english_project_name}
        elif german_project_name:
            filter_dict = {"title": german_project_name}
        
        try:
            if filter_dict:
                docs = kb.similarity_search(query, k=k, filter=filter_dict)
                print(f"Found {len(docs)} documents with metadata filter")
            else:
                docs = kb.similarity_search(query, k=k)
                print(f"Found {len(docs)} documents without filter")
        except Exception as e:
            print(f"Metadata filter failed: {e}, using query-only search")
            docs = kb.similarity_search(query, k=k)
        
        if not docs:
            print(f"No documents found for query: '{query}'")
            return "No relevant information found for this project."
        
        sources = [doc.metadata.get("source", "Unknown") for doc in docs]
        print(f"Retrieved documents: {sources}")
        
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[Source: {source}]\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        print(f"Error in retrieve_context: {e}")
        import traceback
        traceback.print_exc()
        return f"Error retrieving context: {e}"


async def generate_stream(
    chat_id: str,
    user_message: str,
    german_project_name: str,
    english_project_name: str
) -> AsyncGenerator[str, None]:
    dialogue_state, feedback_turn = get_dialogue_state(chat_id)
    
    history = get_chat_history(chat_id)
    last_bot_message = ""
    for msg in reversed(history):
        if msg["role"] == "assistant":
            last_bot_message = msg["content"]
            break
    
    rag_context = retrieve_context(
        user_message, 
        german_project_name=german_project_name,
        english_project_name=english_project_name
    )
    
    intent = await detect_intent(
        message=user_message,
        dialogue_state=dialogue_state,
        feedback_turn=feedback_turn,
        last_bot_message=last_bot_message,
        rag_context=rag_context
    )

    print(f"Detected intent: {intent}")
    if dialogue_state == DialogueState.COMPLETED:
        context = rag_context if intent == 'question' else "No context needed for feedback acknowledgment."
        system_prompt = COSMO_COMPLETED_PROMPT.format(
            german_project_name=german_project_name,
            english_project_name=english_project_name,
            context=context
        )
        new_state = DialogueState.COMPLETED
        new_turn = feedback_turn
    elif intent == 'unsure':
        system_prompt = COSMO_UNSURE_PROMPT.format(
            german_project_name=german_project_name,
            english_project_name=english_project_name,
            last_bot_message=last_bot_message[:300] if last_bot_message else "Welcome message"
        )
        new_state = dialogue_state
        new_turn = feedback_turn
    elif intent == 'question':
        system_prompt = COSMO_QA_PROMPT.format(
            german_project_name=german_project_name,
            english_project_name=english_project_name,
            dialogue_state=dialogue_state,
            feedback_turn=feedback_turn,
            context=rag_context,
            dialogue_starters_en="\n".join(f"- {q}" for q in DIALOGUE_STARTERS),
        )
        new_state = dialogue_state
        new_turn = feedback_turn
    else:
        if dialogue_state == DialogueState.GREETING:
            new_state = DialogueState.GREETING
            new_turn = feedback_turn
        elif dialogue_state == DialogueState.COLLECTING_FEEDBACK:
            new_turn = feedback_turn + 1
            if new_turn >= 3:
                new_state = DialogueState.COMPLETED
            else:
                new_state = DialogueState.COLLECTING_FEEDBACK
        else:
            new_state = dialogue_state
            new_turn = feedback_turn
        
        system_prompt = COSMO_FEEDBACK_PROMPT.format(
            german_project_name=german_project_name,
            english_project_name=english_project_name,
            dialogue_state=new_state,
            feedback_turn=new_turn,
            context=rag_context
        )
    
    messages = [SystemMessage(content=system_prompt)]
    
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    messages.append(HumanMessage(content=user_message))
    save_message(chat_id, "user", user_message)

    full_response = ""
    
    yield f"data: {json.dumps({'intent': intent})}\n\n"
    
    try:
        async for chunk in get_llm().astream(messages):
            if chunk.content:
                full_response += chunk.content
                yield f"data: {json.dumps({'content': chunk.content})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return

    save_message(chat_id, "assistant", full_response)

    if intent != 'question':
        update_dialogue_state(chat_id, new_state, new_turn)
    
    yield f"data: {json.dumps({'done': True, 'dialogue_state': new_state, 'feedback_turn': new_turn})}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_chat_db()
    print(f"Chat history DB: {CHAT_DB_PATH}")
    print(f"Chat model: {CHAT_MODEL}")
    print(f"Chroma DB collection: {CHROMA_DB_COLLECTION_NAME}")

    try:
        ffmpeg_path = check_ffmpeg()
        print(f"FFmpeg found at: {ffmpeg_path}")
    except RuntimeError as e:
        print(f"WARNING: {e}")
        print("Whisper transcription will not work until ffmpeg is installed.")

    print("Initializing Whisper model...")
    get_whisper_model()
    
    print("Initializing OpenAI TTS client...")
    try:
        get_openai_client()
        print("OpenAI TTS ready.")
    except Exception as e:
        print(f"WARNING: OpenAI TTS initialization failed: {e}")
        print("TTS will not work until OPENAI_API_KEY is set correctly.")

    yield
    print("Shutting down...")


app = FastAPI(
    title="Human, Robot! Exhibition Chatbot API",
    description="Interactive guide for the robotics exhibition",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (CSS, images, videos)
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_initial_page():
    """Serve the initial landing page."""
    file_path = os.path.join(FRONTEND_DIR, "initial.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/chat", response_class=HTMLResponse)
async def serve_chat_page():
    """Serve the chat page."""
    file_path = os.path.join(FRONTEND_DIR, "new_chat.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/chats", response_model=CreateChatResponse)
async def create_chat(request: CreateChatRequest):
    chat_id = str(uuid4())
    created_at = datetime.utcnow().isoformat()

    create_chat_record(
        chat_id=chat_id,
        german_name=request.german_project_name,
        english_name=request.english_project_name,
        created_at=created_at
    )
    
    return CreateChatResponse(
        chat_id=chat_id,
        german_project_name=request.german_project_name,
        english_project_name=request.english_project_name,
        created_at=created_at
    )


@app.get("/chats/{chat_id}", response_model=ChatInfo)
async def get_chat(chat_id: str):
    metadata = get_chat_metadata(chat_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    message_count = get_message_count(chat_id)
    dialogue_state, feedback_turn = get_dialogue_state(chat_id)
    
    return ChatInfo(
        chat_id=chat_id,
        german_project_name=metadata.get("german_project_name", ""),
        english_project_name=metadata.get("english_project_name", ""),
        created_at=metadata.get("created_at", ""),
        message_count=message_count,
        dialogue_state=dialogue_state,
        feedback_turn=feedback_turn
    )


@app.post("/chats/{chat_id}/start", response_model=GreetingResponse)
async def start_dialogue(chat_id: str, request: StartDialogueRequest = None):
    metadata = get_chat_metadata(chat_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    dialogue_state, feedback_turn = get_dialogue_state(chat_id)
    
    if dialogue_state != DialogueState.GREETING:
        raise HTTPException(
            status_code=400, 
            detail=f"Dialogue already started. Current state: {dialogue_state}"
        )
    
    language = request.language if request else "en"
    
    if language == "de":
        starters = DIALOGUE_STARTERS_DE
        greeting = "Willkommen bei COSMO! Ich bin hier, um Ihre Gedanken und Fragen zur Ausstellung zu sammeln und an die Forscher weiterzugeben.\n\n"
    elif language == "ru":
        starters = DIALOGUE_STARTERS_RU
        greeting = "Добро пожаловать в COSMO! Я здесь, чтобы собрать ваши мысли и вопросы о выставке и передать их исследователям.\n\n"
    else:
        starters = DIALOGUE_STARTERS
        greeting = "Welcome to COSMO! I'm here to collect your thoughts and questions about the exhibition and pass them on to researchers.\n\n"
    
    starter_question = random.choice(starters)
    full_message = greeting + starter_question
    
    save_message(chat_id, "assistant", full_message)
    
    update_dialogue_state(chat_id, DialogueState.COLLECTING_FEEDBACK, 1)
    
    return GreetingResponse(
        message=full_message,
        dialogue_state=DialogueState.COLLECTING_FEEDBACK,
        feedback_turn=1
    )


@app.post("/chats/{chat_id}/messages")
async def send_message(chat_id: str, request: ChatMessageRequest):
    metadata = get_chat_metadata(chat_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    german_project_name = metadata.get("german_project_name", "")
    english_project_name = metadata.get("english_project_name", "")
    
    return StreamingResponse(
        generate_stream(
            chat_id=chat_id,
            user_message=request.message,
            german_project_name=german_project_name,
            english_project_name=english_project_name
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/chats/{chat_id}/history")
async def get_chat_history_endpoint(chat_id: str):
    metadata = get_chat_metadata(chat_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    history = get_chat_history(chat_id)
    
    return {
        "chat_id": chat_id,
        "messages": history
    }


@app.websocket("/chats/{chat_id}/transcribe/stream")
async def transcribe_audio_stream(websocket: WebSocket, chat_id: str):
    await websocket.accept()
    
    metadata = get_chat_metadata(chat_id)
    if not metadata:
        await websocket.close(code=1008, reason="Chat not found")
        return
    
    try:
        check_ffmpeg()
    except RuntimeError as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close(code=1011, reason="FFmpeg not available")
        return
    
    tmp_path = None
    tmp_fd = None
    audio_chunks = []
    last_transcription_time = asyncio.get_event_loop().time()
    transcription_interval = 2.0
    
    try:
        model = get_whisper_model()
        
        file_extension = ".webm"
        language = None
        
        while True:
            data = await websocket.receive()
            
            if "text" in data:
                message = json.loads(data["text"])
                if message.get("type") == "start":
                    file_extension = message.get("extension", ".webm")
                    language = message.get("language", None)
                    await websocket.send_json({"status": "ready"})
                elif message.get("type") == "end":
                    break
            elif "bytes" in data:
                audio_chunk = data["bytes"]
                if not isinstance(audio_chunk, bytes):
                    audio_chunk = bytes(audio_chunk)
                audio_chunks.append(audio_chunk)
                
                current_time = asyncio.get_event_loop().time()
                if current_time - last_transcription_time >= transcription_interval and len(audio_chunks) > 0:
                    tmp_fd, tmp_path = tempfile.mkstemp(suffix=file_extension)
                    try:
                        with os.fdopen(tmp_fd, 'wb') as tmp_file:
                            for chunk in audio_chunks:
                                tmp_file.write(chunk)
                            tmp_file.flush()
                            os.fsync(tmp_file.fileno())
                        tmp_fd = None
                        
                        tmp_path = os.path.abspath(tmp_path)
                        
                        result = model.transcribe(tmp_path, language=language)
                        transcribed_text = result["text"].strip()
                        
                        if transcribed_text:
                            await websocket.send_json({
                                "type": "partial",
                                "text": transcribed_text,
                                "language": result.get("language", "unknown")
                            })

                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                        tmp_path = None
                        
                        last_transcription_time = current_time
                    except Exception as e:
                        print(f"Error during partial transcription: {e}")
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                        tmp_path = None
        
        if len(audio_chunks) > 0:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=file_extension)
            try:
                with os.fdopen(tmp_fd, 'wb') as tmp_file:
                    for chunk in audio_chunks:
                        tmp_file.write(chunk)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())
                tmp_fd = None
                
                tmp_path = os.path.abspath(tmp_path)
                
                result = model.transcribe(tmp_path, language=language)
                transcribed_text = result["text"].strip()
                
                await websocket.send_json({
                    "type": "final",
                    "text": transcribed_text,
                    "language": result.get("language", "unknown")
                })
                
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception as e:
                print(f"Error during final transcription: {e}")
                import traceback
                traceback.print_exc()
                try:
                    await websocket.send_json({"error": str(e)})
                except:
                    pass
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except:
                pass
        
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        try:
            await websocket.close()
        except:
            pass




class TTSRequest(BaseModel):
    text: str
    language: str = "en"


def split_into_sentences(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


@app.post("/tts/stream")
async def tts_stream(request: TTSRequest):
    
    async def generate():
        try:
            client = get_openai_client()
            voice_name = TTS_VOICES.get(request.language, TTS_VOICES["en"])
            sentences = split_into_sentences(request.text)
            
            if not sentences:
                sentences = [request.text]
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                instructions = """Identity: A robot\n\nAffect: Monotone, mechanical, and neutral, reflecting the robotic nature of the customer service agent.\n\nTone: Efficient, direct, and formal, with a focus on delivering information clearly and without emotion.\n\nEmotion: Neutral and impersonal, with no emotional inflection, as the robot voice is focused purely on functionality.\n\nPauses: Brief and purposeful, allowing for processing and separating key pieces of information, such as confirming the return and refund details.\n\nPronunciation: Clear, precise, and consistent, with each word spoken distinctly to ensure the customer can easily follow the automated process."""

                response = client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice=voice_name,
                    input=sentence,
                    instructions=instructions,
                    response_format="mp3"
                )
                
                # Read audio data
                audio_data = response.content
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                yield f"data: {json.dumps({'sentence': sentence, 'audio': audio_base64})}\n\n"
                await asyncio.sleep(0.01)
                    
        except Exception as e:
            print(f"TTS error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "llm_provider": "openai"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
