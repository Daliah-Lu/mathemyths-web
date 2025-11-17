import os
import time
import tempfile
import datetime
import json
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import dashscope
from dashscope import MultiModalConversation
from openai import OpenAI


# base configuration

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise RuntimeError("先在终端里 export DASHSCOPE_API_KEY=sk-key")

# DashScope SDK alibabaclound singapore
dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

# Qwen 文本模型
llm_client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# sesson state (single user demo)

session_state: Dict[str, Any] = {
    "stage": 0,          
    "substage": 0,       
    "character": None,
    "child_name": None,  
    "story_name": None,  
    "setting": None,
    "terms": ["一半", "等分", "总和", "差", "面积", "估计"],  
    "term_index": 0,
    "max_turns": 6,
    "turn": 0,
    "history": [],       
}

# logs

request_logs: List[Dict[str, Any]] = []
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "session.log")


def ensure_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)


def log_latency(
    asr_ms: float,
    llm_ms: float,
    tts_ms: float,
    total_ms: float,
    asr_text: str = None,
    reply_text: str = None,
):
    """记录每次交互的时延和对话文本，写入内存和本地文件"""
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "asr_ms": round(asr_ms, 1),
        "llm_ms": round(llm_ms, 1),
        "tts_ms": round(tts_ms, 1),
        "total_ms": round(total_ms, 1),
        "stage": session_state["stage"],
        "term_index": session_state["term_index"],
        "turn": session_state["turn"],
        "asr_text": asr_text,
        "reply_text": reply_text,
    }
    request_logs.append(entry)
    if len(request_logs) > 200:
        del request_logs[:-200]

    # 写入文件
    try:
        ensure_log_dir()
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # 文件写入失败不影响主流程
        pass


# session stat and the whole flow

def reset_session():
    session_state.update(
        {
            "stage": 0,
            "substage": 0,
            "character": None,
            "child_name": None,
            "story_name": None,
            "setting": None,
            "term_index": 0,
            "max_turns": 6,
            "turn": 0,
            "history": [],
        }
    )


def generate_teacher_reply(asr_text: str) -> str:
    """
    根据当前状态机，生成老师的中文语音内容。
    整个过程全中文，不用返回文字给前端。
    """
    text = asr_text.strip()
    if not text:
        return "刚才我没有听清楚，你可以再说一遍吗？"

    stage = session_state["stage"]

    # Stage 1: 故事设定（孩子名字、角色、主角名、场景）
    if stage == 1:
        sub = session_state["substage"]

        # 1.0 问名字
        if sub == 0:
            session_state["child_name"] = text[:6]
            session_state["substage"] = 1
            return (
                f"{session_state['child_name']}，欢迎你。接下来我们要一起创造一个故事。"
                "你觉得故事里的主角是什么呢？可以是小兔子、小猫、小汽车，或者任何你喜欢的东西。"
            )

        # 1.1 主角类型
        if sub == 1:
            if "不知道" in text or len(text) < 2:
                session_state["character"] = "小兔子"
            else:
                session_state["character"] = text[:4]
            session_state["substage"] = 2
            return (
                f"好，那我们的主角就是{session_state['character']}。"
                "那它要叫一个什么名字呢？你可以随便起一个好听的名字。"
            )

        # 1.2 主角名字
        if sub == 2:
            if "不知道" in text or len(text) < 2:
                session_state["story_name"] = "乐乐"
            else:
                session_state["story_name"] = text[:4]
            session_state["substage"] = 3
            return (
                f"太好了，我们的{session_state['character']}就叫{session_state['story_name']}。"
                "那故事发生在哪里比较好呢？比如在操场、在家里、在森林里，或者你想到的其他地方。"
            )

        # 1.3 故事场景
        if sub == 3:
            if "不知道" in text or len(text) < 2:
                session_state["setting"] = "森林"
            else:
                session_state["setting"] = text[:6]
            session_state["stage"] = 2
            session_state["substage"] = 0
            return (
                f"好的，那我们的故事设定就是：在{session_state['setting']}，"
                f"有一只{session_state['character']}，它的名字叫{session_state['story_name']}。"
                "接下来，我会一边讲故事，一边带着你留心一些数学词，比如一半、等分、总和、差、面积和估计。"
                "你准备好了吗？如果准备好了，就简单说一句“好了”或者“开始吧”。"
            )

    # Stage 2: 进入故事主线前的过渡
    if stage == 2:
        session_state["stage"] = 3
        session_state["turn"] = 0
        session_state["term_index"] = 0
        return (
            "那我们现在就开始故事的第一段。我讲完之后，会请你补充一点内容。"
            "你可以自由发挥，不用太在意对不对。"
        )

    # Stage 3, 4: 故事主线, 数学词汇循环
    if stage in (3, 4):
        # 如果孩子说得特别少，就先鼓励
        if len(text.split()) < 3 and len(text) < 10:
            system_prompt = (
                "你是一位耐心的儿童数学语言老师。"
                "现在只需要做一件事：用简短、口语化的中文鼓励学生再多说一点，"
                "可以给一个非常简单的提示，不要解释理论，不要讲太长。"
            )
            completion = llm_client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.7,
                max_tokens=120,
            )
            reply = completion.choices[0].message.content.strip()
            return reply

        current_term = None
        if session_state["term_index"] < len(session_state["terms"]):
            current_term = session_state["terms"][session_state["term_index"]]

        system_prompt = (
            "你是一位讲故事的数学语言老师，只用简短的中文跟孩子说话。"
            "你所在的课堂叫“数学语言小课堂”。"
            "故事设定是：在"
            f"{session_state['setting']}，有一只{session_state['character']}，名字叫{session_state['story_name']}。"
            "请按照下面原则回答："
            "1）先用2到3句话继续讲这个故事，把学生刚才说的话自然接进去；"
            "2）如果有当前要练习的数学词，就自然地提到这个词，并用生活化的方式解释一下，不要太抽象；"
            "3）最后用一句简短的问题结尾，邀请孩子再补充一点内容；"
            "4）整体字数控制在80字以内，语气轻松、温和。"
        )
        if current_term:
            system_prompt += f" 当前你要特别带出的数学词是：“{current_term}”。"

        history_text = ""
        for turn in session_state["history"][-4:]:
            history_text += f"{turn['role']}：{turn['content']}\n"

        user_message = (
            "以下是最近几轮对话的简要记录（如果有的话）：\n"
            + history_text
            + "\n这是学生刚才说的话，请你在故事里接住：\n"
            + text
        )

        completion = llm_client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=260,
        )
        reply = completion.choices[0].message.content.strip()

        session_state["history"].append({"role": "user", "content": text})
        session_state["history"].append({"role": "assistant", "content": reply})
        session_state["turn"] += 1

        if current_term is not None:
            session_state["term_index"] += 1

        if (
            session_state["turn"] >= session_state["max_turns"]
            or session_state["term_index"] >= len(session_state["terms"])
        ):
            session_state["stage"] = 5

        return reply

    # Stage 5: 结束
    if stage == 5:
        terms_str = "、".join(session_state["terms"])
        closing = (
            f"今天的数学语言小课堂就到这里啦。我们一起和"
            f"{session_state.get('character') or '主角'}"
            f"{session_state.get('story_name') or ''}讲了一个故事，"
            f"还碰到了这些数学词：{terms_str}。"
            "下次我们可以换一个新的故事，再见。"
        )
        reset_session()
        return closing

    # 兜底
    reset_session()
    return "我们好像有一点点走神了，可以重新从你的名字开始吗？你先告诉我你叫什么。"


def prepare_welcome_text() -> str:
    """
    打开网页时的欢迎语：
    - 重置整个 session 状态
    - 把 stage 设置为 1，让下一次用户说话进入“问名字”阶段
    """
    reset_session()
    session_state["stage"] = 1
    session_state["substage"] = 0

    welcome = (
        "欢迎来到数学语言小课堂。我是你的兔兔老师。"
        "接下来我们会用一个故事，一边说一边学数学词。"
        "你可以这样操作：第一步，等我先说完；第二步，点击中间的圆圈开始说话；"
        "第三步，再点一次圆圈结束录音，然后等我回应你。"
        "现在，你先听我说完，再按圆圈告诉我你叫什么名字。"
    )
    return welcome


# 前端：兔兔老师 + 录音按钮

@app.get("/", response_class=HTMLResponse)
async def index():
    html = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8" />
        <title>数学语言小课堂</title>
        <style>
            * { box-sizing: border-box; }
            body {
                margin: 0;
                padding: 0;
                background: radial-gradient(circle at top, #1f2933, #020617);
                color: #f9fafb;
                font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .container {
                width: 100%;
                max-width: 900px;
                padding: 32px 20px;
            }
            .card {
                background: rgba(15, 23, 42, 0.9);
                border-radius: 24px;
                border: 1px solid rgba(148, 163, 184, 0.4);
                padding: 24px 24px 32px;
                box-shadow: 0 24px 60px rgba(15, 23, 42, 0.9);
            }
            .header {
                display: flex;
                align-items: center;
                gap: 16px;
                margin-bottom: 20px;
            }
            .title-text {
                font-size: 24px;
                font-weight: 600;
                letter-spacing: 0.08em;
            }
            .subtitle {
                font-size: 14px;
                opacity: 0.9;
                margin-top: 4px;
            }
            .teacher-avatar {
                width: 80px;
                height: 80px;
                border-radius: 50%;
                background: radial-gradient(circle at 30% 30%, #f9fafb, #e2e8f0);
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                box-shadow: 0 0 30px rgba(248, 250, 252, 0.6);
            }
            .teacher-avatar svg {
                width: 60px;
                height: 60px;
            }
            .ears {
                position: absolute;
                top: -30px;
                display: flex;
                justify-content: space-between;
                width: 60px;
                left: 10px;
            }
            .ear {
                width: 20px;
                height: 40px;
                border-radius: 999px;
                background: #e5e7eb;
                border: 2px solid #cbd5f5;
            }
            .ear-inner {
                margin: 6px 3px;
                border-radius: 999px;
                background: #f9a8d4;
                height: 28px;
            }
            .main {
                margin-top: 12px;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 18px;
            }
            #circle {
                width: 180px;
                height: 180px;
                border-radius: 50%;
                background: radial-gradient(circle at 30% 30%, #38bdf8, #1d4ed8);
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                box-shadow: 0 0 40px rgba(56, 189, 248, 0.8);
                transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.18s ease;
            }
            #circle.listening {
                transform: scale(1.08);
                box-shadow: 0 0 70px rgba(96, 165, 250, 1);
            }
            #mic-icon {
                width: 70px;
                height: 70px;
            }
            #status {
                font-size: 14px;
                opacity: 0.9;
                min-height: 1.6em;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <div class="header">
                    <div class="teacher-avatar">
                        <div class="ears">
                            <div class="ear"><div class="ear-inner"></div></div>
                            <div class="ear"><div class="ear-inner"></div></div>
                        </div>
                        <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                            <circle cx="32" cy="34" r="20" fill="#e5e7eb" />
                            <circle cx="25" cy="30" r="4" fill="#111827" />
                            <circle cx="39" cy="30" r="4" fill="#111827" />
                            <circle cx="24" cy="29" r="1.2" fill="#f9fafb" />
                            <circle cx="38" cy="29" r="1.2" fill="#f9fafb" />
                            <circle cx="32" cy="36" r="2.4" fill="#0f172a" />
                            <path d="M27 42 Q32 46 37 42" stroke="#0f172a" stroke-width="2.2" stroke-linecap="round" fill="none" />
                        </svg>
                    </div>
                    <div>
                        <div class="title-text">欢迎来到数学语言小课堂</div>
                        <div class="subtitle">兔兔老师会用故事和你一起练习数学词，一切都通过声音来完成。</div>
                    </div>
                </div>

                <div class="main">
                    <div id="circle" onclick="toggleRecording()">
                        <svg id="mic-icon" viewBox="0 0 24 24" fill="none">
                            <rect x="9" y="4" width="6" height="10" rx="3" stroke="white" stroke-width="1.6"/>
                            <path d="M6 11a1 1 0 0 0 1 1h0a5 5 0 0 0 10 0h0a1 1 0 0 0 1-1" stroke="white" stroke-width="1.6" stroke-linecap="round"/>
                            <path d="M12 17v3" stroke="white" stroke-width="1.6" stroke-linecap="round"/>
                            <path d="M9 20h6" stroke="white" stroke-width="1.6" stroke-linecap="round"/>
                        </svg>
                    </div>
                    <div id="status">兔兔老师正在准备中，请稍等片刻。</div>
                </div>
            </div>
        </div>

        <script>
            let mediaRecorder = null;
            let audioChunks = [];
            let isRecording = false;

            function setStatus(text) {
                document.getElementById("status").innerText = text;
            }

            async function toggleRecording() {
                if (!isRecording) {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        mediaRecorder = new MediaRecorder(stream);
                        audioChunks = [];

                        mediaRecorder.ondataavailable = (event) => {
                            if (event.data.size > 0) {
                                audioChunks.push(event.data);
                            }
                        };

                        mediaRecorder.onstop = async () => {
                            setStatus("正在思考，请稍等...");
                            const blob = new Blob(audioChunks, { type: "audio/webm" });
                            await sendAudio(blob);
                            stream.getTracks().forEach(t => t.stop());
                        };

                        mediaRecorder.start();
                        isRecording = true;
                        document.getElementById("circle").classList.add("listening");
                        setStatus("录音中，再点一次结束。");
                    } catch (err) {
                        console.error(err);
                        setStatus("无法访问麦克风，请检查浏览器的麦克风权限。");
                    }
                } else {
                    if (mediaRecorder) {
                        mediaRecorder.stop();
                    }
                    isRecording = false;
                    document.getElementById("circle").classList.remove("listening");
                    setStatus("正在上传音频...");
                }
            }

            async function sendAudio(blob) {
                try {
                    const formData = new FormData();
                    formData.append("audio", blob, "input.webm");

                    const response = await fetch("/api/voice", {
                        method: "POST",
                        body: formData
                    });

                    if (!response.ok) {
                        setStatus("后端出错，请稍后重试。");
                        return;
                    }

                    const data = await response.json();
                    if (data.error) {
                        setStatus(data.error);
                        return;
                    }

                    if (data.audio_url) {
                        setStatus("等兔兔老师说完，你就可以再点圆圈继续说话。");
                        const audio = new Audio(data.audio_url);
                        audio.play();
                    } else {
                        setStatus("没有拿到语音回复。");
                    }
                } catch (e) {
                    console.error(e);
                    setStatus("网络错误，请稍后重试。");
                }
            }

            async function playWelcome() {
                try {
                    setStatus("兔兔老师正在打招呼，请稍等...");
                    const res = await fetch("/api/welcome");
                    if (!res.ok) {
                        setStatus("欢迎语加载失败，你可以直接点击圆圈开始说话。");
                        return;
                    }
                    const data = await res.json();
                    if (data.audio_url) {
                        const audio = new Audio(data.audio_url);
                        audio.play();
                        setStatus("等兔兔老师说完，再点击圆圈开始说话，再点一次结束录音。");
                    } else {
                        setStatus("欢迎语没有返回音频，你可以直接点击圆圈说话。");
                    }
                } catch (e) {
                    console.error(e);
                    setStatus("欢迎语加载出错，你可以直接点击圆圈开始说话。");
                }
            }

            window.addEventListener("load", () => {
                playWelcome();
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


# 欢迎语接口

@app.get("/api/welcome")
async def play_welcome():
    """
    打开网页时调用：
    - 准备欢迎语
    - 用 TTS 合成为语音
    - 返回音频 URL
    """
    text = prepare_welcome_text()
    try:
        tts_response = dashscope.MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model="qwen3-tts-flash",
            text=text,
            voice="Cherry",
            language_type="Chinese",
        )
        audio_url = None
        if (
            hasattr(tts_response, "output")
            and hasattr(tts_response.output, "audio")
        ):
            audio_url = getattr(tts_response.output.audio, "url", None)

        if not audio_url:
            return JSONResponse({"error": "欢迎语 TTS 没有返回音频 URL"}, status_code=500)

        return JSONResponse({"audio_url": audio_url})
    except Exception as e:
        return JSONResponse({"error": f"欢迎语 TTS 出错: {e}"}, status_code=500)


# 语音交互接口：ASR -> 教学逻辑 -> TTS

@app.post("/api/voice")
async def voice_interaction(audio: UploadFile = File(...)):
    start_total = time.time()

    if not audio.content_type.startswith("audio/"):
        return JSONResponse({"error": "请上传音频文件"}, status_code=400)

    # 1. 保存上传的音频到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        audio_bytes = await audio.read()
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # 2. ASR：Qwen3-ASR-Flash
    start_asr = time.time()
    asr_messages = [
        {
            "role": "system",
            "content": [{"text": ""}],
        },
        {
            "role": "user",
            "content": [{"audio": tmp_path}],
        },
    ]

    asr_text = ""
    try:
        asr_response = MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model="qwen3-asr-flash",
            messages=asr_messages,
            result_format="message",
            asr_options={
                "enable_itn": True,
                "enable_lid": True,
            },
        )

        if (
            hasattr(asr_response, "status_code")
            and asr_response.status_code == 200
            and hasattr(asr_response, "output")
            and hasattr(asr_response.output, "choices")
            and len(asr_response.output.choices) > 0
        ):
            choice = asr_response.output.choices[0]
            if (
                hasattr(choice, "message")
                and hasattr(choice.message, "content")
                and len(choice.message.content) > 0
            ):
                content = choice.message.content[0]
                if isinstance(content, dict) and "text" in content:
                    asr_text = content["text"]
    except Exception as e:
        return JSONResponse({"error": f"ASR 出错: {e}"}, status_code=500)
    asr_ms = (time.time() - start_asr) * 1000

    if not asr_text:
        reply_text = "我刚刚没有听清楚，你可以再说一遍吗？"
        llm_ms = 0.0
    else:
        # 3. 教学逻辑：根据状态机生成中文回复
        start_llm = time.time()
        try:
            reply_text = generate_teacher_reply(asr_text)
        except Exception:
            reply_text = "现在系统有一点点小问题，不过你可以先想一想，等一会儿再和我说一遍。"
        llm_ms = (time.time() - start_llm) * 1000

    # 4. TTS：Qwen3-TTS-Flash
    start_tts = time.time()
    try:
        tts_response = dashscope.MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model="qwen3-tts-flash",
            text=reply_text,
            voice="Cherry",
            language_type="Chinese",
        )
        audio_url = None
        if (
            hasattr(tts_response, "output")
            and hasattr(tts_response.output, "audio")
        ):
            audio_url = getattr(tts_response.output.audio, "url", None)

        if not audio_url:
            return JSONResponse({"error": "TTS 没有返回音频 URL"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": f"TTS 出错: {e}"}, status_code=500)
    tts_ms = (time.time() - start_tts) * 1000

    total_ms = (time.time() - start_total) * 1000
    log_latency(asr_ms, llm_ms, tts_ms, total_ms, asr_text=asr_text, reply_text=reply_text)

    return JSONResponse({"audio_url": audio_url})


# 查看时延日志

@app.get("/logs")
async def get_logs(limit: int = 50):
    """
    返回最近若干条时延日志，默认 50 条。
    访问 http://127.0.0.1:8000/logs 或 /logs?limit=10
    """
    if limit <= 0:
        limit = 1
    data = request_logs[-limit:]
    return {"count": len(data), "logs": data}
