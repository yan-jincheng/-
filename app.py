import streamlit as st
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI

st.set_page_config(
    page_title="文以辨心",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    body, .stApp {
        background-color: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    header[data-testid="stHeader"] {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    .main-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.04);
        padding: 2rem 2rem 1.5rem;
        margin: 1rem auto;
        max-width: 680px;
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
        margin-bottom: 2rem;
    }

    .stTextArea textarea {
        background: #ffffff;
        border: 1px solid #e5e5ea;
        border-radius: 16px;
        padding: 1rem;
        font-size: 1rem;
        color: #1d1d1f;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        transition: all 0.2s;
    }
    .stTextArea textarea:focus {
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
        margin-top: 0.5rem;
        width: 100%;
    }
    div.stButton > button:hover {
        background: #0077ed;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,113,227,0.25);
    }
    div.stButton > button:active {
        transform: scale(0.98);
    }

    .stExpander {
        background: #ffffff;
        border: 1px solid #f0f0f5;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        margin-bottom: 0.75rem;
    }

    .stDownloadButton button {
        background: #f5f5f7;
        color: #0071e3;
        border: 1px solid #e5e5ea;
        border-radius: 12px;
        font-weight: 500;
    }
    .stDownloadButton button:hover {
        background: #ffffff;
    }

    .stAlert {
        border-radius: 14px;
        background: rgba(255,255,255,0.9);
    }

    .stImage, .stPyplot {
        border-radius: 16px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="title-text">🧠 文以辨心</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">AI · 心理学 — 用文字看清你的性格轮廓</div>', unsafe_allow_html=True)

user_text = st.text_area(
    "写下你此刻的想法、心情或任何一段文字",
    height=160,
    placeholder="比如：最近工作节奏很快，但我开始享受这种充实感，周末也想去学一点新东西，认识些有趣的人…"
)

SYSTEM_PROMPT = """你是一位专业的心理学专家，擅长通过文本分析大五人格（OCEAN）。
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
}
"""

api_key = st.secrets["DEEPSEEK_API_KEY"]

def analyze_personality(text):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"文本：{text}"}
        ],
        temperature=0.3,
        max_tokens=500
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

col1, col2, col3 = st.columns([1,2,1])
with col2:
    analyze_btn = st.button("🔍 开始分析", use_container_width=True)

if analyze_btn:
    if not user_text or len(user_text) < 30:
        st.warning("请至少输入30个字，才能获得更准确的分析结果")
    else:
        with st.spinner("AI 正在解读你的文字…"):
            data = analyze_personality(user_text)
        if data:
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

st.markdown('</div>', unsafe_allow_html=True)
