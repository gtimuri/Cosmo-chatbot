# ChatBot Usage

Study project: a simple chatbot with a web interface that works with a local knowledge base.  
The project was implemented within the framework of university assignments.

<table align="center" style="border: none;">
  <tr>
    <td align="center" style="border: none;">
      <img src="https://github.com/user-attachments/assets/1d63b0e8-1fcf-42f8-8d96-78747f5bbb64" alt="ChatBot Interface" width="400">
      <br>
      <br>
      <em>Interactive Chat Interface: Q&A with local knowledge base</em>
    </td>
  </tr>
</table>

---

#### 1. Create pip environment

```bash
python -m venv .venv
source ./.venv/Scripts/activate[.bat]
```

# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Mac/Linux
python -m venv .venv
source .venv/bin/activate

#### 2. Install dependencies

**Python requirements:**
```bash
pip install -r requirements.txt
```

**Ffmpeg:**
* Windows: 
```bash
choco install ffmpeg
``` 
or download from https://ffmpeg.org/download.html

* Linux: 
```bash
sudo apt install ffmpeg
```

* MacOS:
```bash
brew install ffmpeg
```

#### 3. Start Docker Compose (for ChromaDB and Ollama only)

```bash
docker compose up -d
```

#### 4. Upload knowledgebase

```bash
python chunk_and_upload_knowledgebase.py
```

#### 5. Setup .env file

* Copy .env.example end rename to .env
* Fill OPENAI_API_KEY

#### 6. Run the application

```bash
python api.py
```

Service will be available on `http://localhost:8080` by default
