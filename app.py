import streamlit as st
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI

st.set_page_config(page_title="文以辨心", page_icon="🧠", layout="centered", initial_sidebar_state="collapsed")

# ---------- 全局 Apple 风格 + 隐藏默认动画 ----------
st.markdown("""
<style>
    body, .stApp {
        background-color: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    header[data-testid="stHeader"] {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    /* 隐藏右上角运行状态圈 */
    .stStatusWidget {display: none !important;}
    /* 隐藏部署按钮 */
    .stDeployButton {display: none !important;}

    .main-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.04);
        padding: 2rem 2rem 1.5rem;
        margin: 1rem auto;
        max-width: 700px;
    }
    .title-text {
        font-size: 2.5rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #1d1d1f;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .subtitle-text {
        font-size: 1.1rem;
        font-weight: 400;
        color: #86868b;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stTextArea textarea, .stTextInput input {
        background: #ffffff;
        border: 1px solid #e5e5ea;
        border-radius: 16px;
        padding: 0.8rem 1rem;
        font-size: 1rem;
        color: #1d1d1f;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .stTextInput input:focus {
        border-color: #0071e3;
        box-shadow: 0 0 0 3px rgba(0,113,227,0.15);
    }
    div.stButton > button {
        background: #0071e3;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 500;
        transition: all 0.2s;
        width: 100%;
    }
    div.stButton > button:hover {
        background: #0077ed;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,113,227,0.25);
    }
    .chat-bubble {
        padding: 0.8rem 1rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        max-width: 85%;
        line-height: 1.5;
        font-size: 0.95rem;
    }
    .user-bubble {
        background: #0071e3;
        color: white;
        margin-left: auto;
    }
    .ai-bubble {
        background: #f2f2f7;
        color: #1d1d1f;
    }
    .typing-bubble {
        background: #f2f2f7;
        color: #86868b;
        font-style: italic;
        padding: 0.6rem 1rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        max-width: 60%;
    }
    .stExpander {
        background: #ffffff;
        border: 1px solid #f0f0f5;
        border-radius: 16px;
    }
    .stDownloadButton button {
        background: #f5f5f7;
        color: #0071e3;
        border: 1px solid #e5e5ea;
        border-radius: 12px;
    }
    .send-hint {
        font-size: 0.8rem;
        color: #86868b;
        margin-top: 0.2rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 读取 DeepSeek 配置 ----------
if "DEEPSEEK_API_KEY" not in st.secrets:
    st.error("请在 Streamlit Cloud 的 Settings → Secrets 中添加 DEEPSEEK_API_KEY")
    st.stop()

api_key = st.secrets["DEEPSEEK_API_KEY"]
base_url = st.secrets.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
model = st.secrets.get("DEEPSEEK_MODEL", "deepseek-chat")
client = OpenAI(api_key=api_key, base_url=base_url)

# ---------- 工具函数 ----------
def analyze_personality(text):
    system_prompt = """你是一位专业的心理学专家，擅长通过文本分析大五人格（OCEAN）。
请严格按以下要求分析：
1. 开放性(O)：想象力、审美、情感丰富、尝新
2. 尽责性(C)：条理、自律、责任感、目标导向
3. 外向性(E)：社交、热情、积极情绪、活力
4. 宜人性(A)：信任、利他、合作、谦虚
5. 神经质(N)：焦虑、敌意、冲动、脆弱
返回纯JSON，不要有额外文字，格式如下：
{
  "开放性": {"score": 7, "reason": "引用的原文关键词句"},
  "尽责性": {"score": 5, "reason": "..."},
  "外向性": {"score": 3, "reason": "..."},
  "宜人性": {"score": 8, "reason": "..."},
  "神经质": {"score": 4, "reason": "..."}
}"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"文本：{text}"}
        ],
        temperature=0.3, max_tokens=500
    )
    result = response.choices[0].message.content
    json_match = re.search(r'\{.*\}', result, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    else:
        st.error("AI 返回格式异常，请稍后重试")
        return None

def draw_radar(scores):
    labels = list(scores.keys())
    values = [scores[k]["score"] for k in labels]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#f5f5f7')
    ax.set_facecolor('#f5f5f7')
    ax.fill(angles, values, color='#0071e3', alpha=0.15)
    ax.plot(angles, values, color='#0071e3', linewidth=2.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color='#1d1d1f')
    ax.set_ylim(0, 10)
    ax.set_yticks([2,4,6,8,10])
    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.grid(color='#e5e5ea', linewidth=0.8)
    plt.tight_layout()
    return fig

# ---------- 会话状态 ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "result" not in st.session_state:
    st.session_state.result = None

# ---------- 页面主卡片 ----------
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="title-text">🧠 文以辨心</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">AI · 心理学 — 用文字看清你的性格轮廓</div>', unsafe_allow_html=True)

mode = st.radio("选择模式", ["💬 对话引导（AI会提问，慢慢了解你）", "⚡ 直接分析（输入一段文字即可）"], horizontal=True)

if "对话" in mode:
    # ---------- 对话模式（无旋转动画） ----------
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        elif msg.get("temp"):
            # 临时“对方正在输入…”消息
            st.markdown(f'<div class="typing-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble ai-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        greeting = "嗨！我想更了解你。随便说点什么吧，比如最近的心情、让你开心或烦恼的事…"
        st.session_state.messages.append({"role": "ai", "content": greeting})
        st.rerun()

    # 输入表单（回车或发送按钮均可）
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "输入消息",
            placeholder="在这里打字… 按 Enter 发送",
            label_visibility="collapsed"
        )
        st.markdown('<div class="send-hint">↵ 按回车发送消息</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            send_btn = st.form_submit_button("发送")
        with col2:
            analyze_chat_btn = st.form_submit_button("🔍 分析对话")

    # 处理发送
    if send_btn and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "ai", "content": "对方正在输入…", "temp": True})
        st.rerun()

    # 检测是否需要生成 AI 回复（临时消息存在）
    if st.session_state.messages and st.session_state.messages[-1].get("temp"):
        # 移除临时消息
        st.session_state.messages.pop()
        clean_msgs = [m for m in st.session_state.messages if not m.get("temp")]
        convo_text = "\n".join([f"{m['role']}: {m['content']}" for m in clean_msgs])
        follow_up = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个温暖、善于倾听的心理学对话伙伴。基于用户的回答，提出一个简短、开放性的问题，引导ta说出更多内心想法。问题不要超过两句话，语气像朋友聊天。"},
                {"role": "user", "content": f"对话记录：\n{convo_text}\n\n请提出一个后续问题，鼓励用户继续分享。"}
            ],
            temperature=0.8, max_tokens=100
        )
        reply = follow_up.choices[0].message.content.strip()
        st.session_state.messages.append({"role": "ai", "content": reply})
        st.rerun()

    # 分析对话
    if analyze_chat_btn:
        full_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages if not m.get("temp")])
        if len(full_text) < 20:
            st.warning("对话内容太少，再多聊两句吧～")
        else:
            with st.spinner("AI 正在分析你们的对话…"):
                data = analyze_personality(full_text)
            if data:
                st.session_state.result = data
                st.session_state.analysis_done = True
                st.rerun()

else:
    # ---------- 直接分析模式 ----------
    direct_text = st.text_area("写下你此刻的想法、心情或任何一段文字", height=160,
                               placeholder="比如：最近工作节奏很快，但我开始享受这种充实感…")
    if st.button("🔍 开始分析"):
        if len(direct_text) < 10:
            st.warning("至少输入 10 个字，才能获得准确分析")
        else:
            with st.spinner("AI 正在解读你的文字…"):
                data = analyze_personality(direct_text)
            if data:
                st.session_state.result = data
                st.session_state.analysis_done = True
                st.rerun()

# ---------- 展示分析结果 ----------
if st.session_state.analysis_done and st.session_state.result:
    data = st.session_state.result
    st.success("分析完成！")
    fig = draw_radar(data)
    st.pyplot(fig)

    st.markdown("### 维度详解")
    for trait, info in data.items():
        with st.expander(f"{trait}  {info['score']}/10"):
            st.markdown(f"**原文依据**：{info['reason']}")

    st.download_button(
        label="📥 下载 JSON 结果",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name="personality_result.json",
        mime="application/json"
    )

    if st.button("🔄 重新开始"):
        st.session_state.messages = []
        st.session_state.analysis_done = False
        st.session_state.result = None
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
