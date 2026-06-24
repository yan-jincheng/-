import streamlit as st
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from openai import OpenAI
import io
import os
import pickle
import matplotlib.font_manager as fm

# ---------- 持久化存储配置 ----------
HISTORY_FILE = "history_data.pkl"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return []
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, "wb") as f:
            pickle.dump(history, f)
    except Exception as e:
        st.warning(f"数据持久化存储失败: {str(e)}")

# ---------- 页面基本配置 ----------
st.set_page_config(page_title="文以辨心", page_icon="🧠", layout="centered", initial_sidebar_state="collapsed")

# ---------- 中文字体配置（终极优化：防乱码英文回退） ----------
def get_font_or_fallback():
    """
    检查系统是否有中文字体，若没有则返回 'sans-serif'。
    在绘图时，如果无法渲染中文，将自动替换为英文标签。
    """
    font_names = ['PingFang SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    for font in font_names:
        try:
            if fm.findfont(font):
                return font
        except:
            continue
    return 'sans-serif'

plt.rcParams['font.sans-serif'] = [get_font_or_fallback()]
plt.rcParams['axes.unicode_minus'] = False

# ---------- Apple 风格 CSS ----------
st.markdown("""
<style>
    body, .stApp { background-color: #f5f5f7; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    header[data-testid="stHeader"] {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .stStatusWidget {display: none !important;}
    .stDeployButton {display: none !important;}
    .main-card {
        background: rgba(255, 255, 255, 0.8); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
        border-radius: 24px; box-shadow: 0 8px 32px rgba(0,0,0,0.04); padding: 2rem; margin: 1rem auto; max-width: 750px;
    }
    .title-text { font-size: 2.5rem; font-weight: 600; color: #1d1d1f; text-align: center; margin-bottom: 0.25rem; }
    .subtitle-text { font-size: 1.1rem; color: #86868b; text-align: center; margin-bottom: 1.5rem; }
    .stTextArea textarea, .stTextInput input {
        background: #fff; border: 1px solid #e5e5ea; border-radius: 16px; padding: 0.8rem 1rem; font-size: 1rem;
    }
    div.stButton > button {
        background: #0071e3; color: white; border: none; border-radius: 12px; padding: 0.6rem 1.5rem; font-weight: 500; width: 100%;
    }
    div.stButton > button:hover { background: #0077ed; }
    .chat-bubble { padding: 0.8rem 1rem; border-radius: 18px; margin: 0.5rem 0; max-width: 85%; }
    .user-bubble { background: #0071e3; color: white; margin-left: auto; }
    .ai-bubble { background: #f2f2f7; color: #1d1d1f; }
    .typing-bubble { background: #f2f2f7; color: #86868b; font-style: italic; padding: 0.6rem 1rem; border-radius: 18px; margin: 0.5rem 0; max-width: 60%; }
    .stExpander { background: #ffffff; border: 1px solid #f0f0f5; border-radius: 16px; }
    .send-hint { font-size: 0.8rem; color: #86868b; margin-top: 0.2rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ---------- 读取配置 ----------
if "DEEPSEEK_API_KEY" not in st.secrets:
    st.error("请在 Streamlit Cloud 的 Settings → Secrets 中添加 DEEPSEEK_API_KEY")
    st.stop()

client = OpenAI(
    api_key=st.secrets["DEEPSEEK_API_KEY"],
    base_url=st.secrets.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)
model = st.secrets.get("DEEPSEEK_MODEL", "deepseek-chat")

# ---------- 核心工具函数 ----------
def analyze_personality(text):
    """分析大五人格 (强错误回显)"""
    system_prompt = """你是一位专业的心理学专家，擅长通过文本分析大五人格。
请基于文本对以下五个正向维度进行评分（1-10分，分数越高特质越强）：
1. 开放心态：喜欢新体验、想象力丰富、审美敏锐
2. 认真负责：条理清晰、自律可靠、目标驱动
3. 社交外向：热情健谈、喜欢群体活动、积极情绪
4. 同理合作：信任他人、乐于助人、重视和谐
5. 情绪稳定：情绪平稳、抗压能力强、不易焦虑
返回纯JSON，不要有任何额外文字，格式如下：
{
  "开放心态": {"score": 7, "reason": "引用的原文关键词句"},
  "认真负责": {"score": 5, "reason": "..."},
  "社交外向": {"score": 3, "reason": "..."},
  "同理合作": {"score": 8, "reason": "..."},
  "情绪稳定": {"score": 4, "reason": "..."}
}"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"文本：{text}"}],
            temperature=0.3, max_tokens=500
        )
        result = response.choices[0].message.content
        
        # 找到大模型回复中的 JSON 部分
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if not json_match:
            st.error(f"❌ 大模型返回格式异常，未找到JSON。返回原始内容：\n{result}")
            return None
        
        json_data = json.loads(json_match.group())
        return json_data
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON解析失败，请检查大模型输出格式。错误详情：{str(e)}\n原始内容片段：{result[:200]}")
        return None
    except Exception as e:
        st.error(f"❌ API调用发生致命错误：{str(e)}")
        return None

def summarize_text(text):
    """对长对话进行智能压缩，节省 Token 消耗"""
    if len(text) < 1000:
        return text
    try:
        prompt = f"将以下心理对话文本，压缩摘要成200字以内的核心情感与内容摘要，保留用户表达的真实情绪，不要丢失细节：\n\n{text}"
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5, max_tokens=300
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"⚠️ 摘要生成失败，将使用原文本分析: {str(e)}")
        return text

# ---------- 核心绘图函数 ----------
# 如果检测不到中文字体，直接使用英文标签，杜绝方块
LABEL_MAP = {
    "开放心态": "Openness",
    "认真负责": "Conscientiousness",
    "社交外向": "Extraversion",
    "同理合作": "Agreeableness",
    "情绪稳定": "Emotional Stability"
}

def draw_radar(scores, title=None, color='#0071e3', ax=None):
    labels = list(scores.keys())
    values = [scores[k]["score"] for k in labels]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor('#f5f5f7')
        ax.set_facecolor('#fafafa')
    else:
        fig = ax.figure

    ax.fill(angles, values, color=color, alpha=0.15)
    ax.plot(angles, values, color=color, linewidth=3, marker='o', markersize=9,
            markerfacecolor='white', markeredgecolor=color, markeredgewidth=3)
    
    ax.set_ylim(0, 11)
    ax.set_rticks([2, 4, 6, 8, 10])  
    ax.set_rlabel_position(45)      
    
    ax.grid(True, linestyle='--', color='#d1d1d6', linewidth=0.8, alpha=0.8)
    ax.spines['polar'].set_visible(False)

    # 针对每个标签进行极坐标绘制
    for label, angle in zip(labels, angles[:-1]):
        rotation = np.rad2deg(angle)
        if 90 <= rotation <= 270:
            ha = 'right'
        else:
            ha = 'left'
        if 0 <= rotation <= 180:
            va = 'bottom'
        else:
            va = 'top'
        
        # 检查当前使用的字体是否支持中文
        if plt.rcParams['font.sans-serif'][0] == 'sans-serif':
            # 如果无法找到中文字体，自动替换为英文，防止出现方块
            display_label = LABEL_MAP.get(label, label)
        else:
            display_label = label

        ax.text(angle, 10.8, display_label, ha=ha, va=va, 
                fontsize=13, fontweight='500', color='#1d1d1f')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', color='#1d1d1f', pad=25)
    return fig

def score_label(score):
    if score >= 8: return "较高"
    elif score >= 5: return "适中"
    else: return "稍低"

def generate_card(scores):
    fig, (ax_radar, ax_text) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1.3, 1]})
    fig.patch.set_facecolor('#f5f5f7')
    draw_radar(scores, ax=ax_radar)
    ax_text.axis('off')
    traits = [f"{k} {v['score']}/10 {score_label(v['score'])}" for k, v in scores.items()]
    summary = "\n".join(traits)
    ax_text.text(0.5, 0.75, "🧠 文以辨心", fontsize=22, fontweight='bold', color='#1d1d1f',
                 ha='center', va='center', transform=ax_text.transAxes)
    ax_text.text(0.5, 0.45, summary, fontsize=13, color='#86868b', ha='center', va='center',
                 transform=ax_text.transAxes, linespacing=1.8)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    ax_text.text(0.5, 0.15, f"分析时间：{now}", fontsize=9, color='#b0b0b5', ha='center', va='center',
                 transform=ax_text.transAxes)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight') 
    buf.seek(0)
    return buf

# ---------- 会话状态 (初始化本地持久化历史) ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "result" not in st.session_state:
    st.session_state.result = None
if "history" not in st.session_state:
    st.session_state.history = load_history()
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "💬 对话引导"
if "compare_list" not in st.session_state:
    st.session_state.compare_list = []

# ---------- 页面主卡片 ----------
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="title-text">🧠 文以辨心</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">AI · 心理学 — 用文字看清你的性格轮廓</div>', unsafe_allow_html=True)

mode = st.radio("选择模式", ["💬 对话引导", "⚡ 直接分析", "📊 历史对比", "🖼️ 分享卡片"], horizontal=True,
                key="mode_radio")
st.session_state.selected_mode = mode

if mode == "💬 对话引导":
    # 显示历史消息
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        elif msg.get("temp"):
            st.markdown(f'<div class="typing-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble ai-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        greeting = "嗨！我想更了解你。随便说点什么吧，比如最近的心情、让你开心或烦恼的事…"
        st.session_state.messages.append({"role": "ai", "content": greeting})
        st.rerun()

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("输入消息", placeholder="在这里打字… 按 Enter 发送", label_visibility="collapsed")
        st.markdown('<div class="send-hint">↵ 按回车发送消息</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            send_btn = st.form_submit_button("发送")
        with col2:
            analyze_chat_btn = st.form_submit_button("🔍 分析对话")

    if send_btn and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "ai", "content": "对方正在输入…", "temp": True})
        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1].get("temp"):
        st.session_state.messages.pop()
        clean_msgs = [m for m in st.session_state.messages if not m.get("temp")]
        convo_text = "\n".join([f"{m['role']}: {m['content']}" for m in clean_msgs])
        with st.spinner("正在思考…"):
            try:
                follow_up = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是一个温暖、善于倾听的心理学对话伙伴。基于用户的回答，提出一个简短、开放性的问题，引导ta说出更多内心想法。问题不要超过两句话，语气像朋友聊天。"},
                        {"role": "user", "content": f"对话记录：\n{convo_text}\n\n请提出一个后续问题，鼓励用户继续分享。"}
                    ],
                    temperature=0.8, max_tokens=100
                )
                reply = follow_up.choices[0].message.content.strip()
            except Exception as e:
                reply = f"（抱歉，我这边接收信息卡了一下，你可以再和我聊聊吗？错误提示：{str(e)}）"
        st.session_state.messages.append({"role": "ai", "content": reply})
        st.rerun()

    if analyze_chat_btn:
        full_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages if not m.get("temp")])
        if len(full_text) < 20:
            st.warning("对话内容太少，再多聊两句吧～")
        else:
            with st.spinner("AI 正在分析你们的对话（长对话会自动压缩以节省Token）…"):
                # 1. 压缩文本
                analysis_text = summarize_text(full_text)
                # 2. 分析人格
                data = analyze_personality(analysis_text)
            if data:
                st.session_state.result = data
                st.session_state.analysis_done = True
                record = {
                    "time": datetime.now().strftime("%m-%d %H:%M"),
                    "text": full_text[:60] + ("..." if len(full_text)>60 else ""),
                    "result": data
                }
                st.session_state.history.insert(0, record)
                if len(st.session_state.history) > 5:
                    st.session_state.history.pop()
                save_history(st.session_state.history)
                st.rerun()

elif mode == "⚡ 直接分析":
    direct_text = st.text_area("写下你此刻的想法、心情或任何一段文字", height=160,
                               placeholder="比如：最近工作节奏很快，但我开始享受这种充实感，也想去学新东西…")
    if st.button("🔍 开始分析"):
        if len(direct_text) < 10:
            st.warning("至少输入10个字，才能获得准确分析")
        else:
            with st.spinner("AI 正在解读…"):
                data = analyze_personality(direct_text)
            if data:
                st.session_state.result = data
                st.session_state.analysis_done = True
                record = {
                    "time": datetime.now().strftime("%m-%d %H:%M"),
                    "text": direct_text[:60] + ("..." if len(direct_text)>60 else ""),
                    "result": data
                }
                st.session_state.history.insert(0, record)
                if len(st.session_state.history) > 5:
                    st.session_state.history.pop()
                save_history(st.session_state.history)
                st.rerun()

elif mode == "📊 历史对比":
    if not st.session_state.history:
        st.info("暂无历史记录，先去分析吧～")
    else:
        if st.button("🗑️ 清空所有历史记录 (慎重操作)"):
            st.session_state.history = []
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()
        
        st.markdown("### 📜 最近五次分析")
        for i, item in enumerate(st.session_state.history):
            col1, col2 = st.columns([4,1])
            with col1:
                st.write(f"**{item['time']}**：{item['text']}")
            with col2:
                if st.button("选择", key=f"sel_{i}"):
                    if len(st.session_state.compare_list) < 2:
                        st.session_state.compare_list.append(item)
                    else:
                        st.session_state.compare_list = [st.session_state.compare_list[-1], item]
                    st.rerun()
        if len(st.session_state.compare_list) == 2:
            st.markdown("### ⚖️ 对比雷达图")
            try:
                fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
                fig.patch.set_facecolor('#f5f5f7')
                ax.set_facecolor('#fafafa')
                colors = ['#0071e3', '#ff3b30']
                all_labels = list(st.session_state.compare_list[0]['result'].keys())
                angles = np.linspace(0, 2*np.pi, len(all_labels), endpoint=False).tolist()
                angles += angles[:1]
                for idx, item in enumerate(st.session_state.compare_list):
                    scores = item['result']
                    values = [scores[k]["score"] for k in all_labels]
                    values += values[:1]
                    ax.fill(angles, values, color=colors[idx], alpha=0.1)
                    ax.plot(angles, values, color=colors[idx], linewidth=2.5, marker='o', markersize=6,
                            markerfacecolor='white', markeredgecolor=colors[idx], markeredgewidth=2,
                            label=f"{item['time']} {item['text'][:10]}")
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(all_labels, fontsize=10)
                ax.set_ylim(0,10)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                ax.grid(color='#d1d1d6', linestyle='--', linewidth=0.6)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"对比图生成出错: {str(e)}")
            
            if st.button("清除对比"):
                st.session_state.compare_list = []
                st.rerun()

elif mode == "🖼️ 分享卡片":
    if st.session_state.result:
        st.success("分析结果已就绪")
        if st.button("📸 生成分享卡片"):
            buf = generate_card(st.session_state.result)
            st.download_button("⬇️ 下载 PNG", data=buf, file_name="personality_card.png", mime="image/png")
    else:
        st.warning("请先完成一次人格分析")

# ---------- 展示分析结果 ----------
if st.session_state.analysis_done and st.session_state.result:
    st.markdown("---")
    st.success("分析完成！")
    
    # 检查字体适配情况并在前端做提示
    current_font = plt.rcParams['font.sans-serif'][0]
    if current_font == 'sans-serif':
        st.info("💡 友情提示：当前云服务器环境未检测到中文字体。为了保证图表正常显示，雷达图标签已自动切换为英文标准术语 (Openness, Conscientiousness...)。下方维度的中文解释与分数对应不受影响。")

    fig = draw_radar(st.session_state.result, title="人格轮廓")
    st.pyplot(fig)

    st.markdown("### 维度详解")
    for trait, info in st.session_state.result.items():
        label = score_label(info['score'])
        with st.expander(f"{trait}  {info['score']}/10 ({label})"):
            st.write(f"**原文依据**：{info['reason']}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧠 生成深度解读报告"):
            with st.spinner("AI 正在撰写报告…"):
                summary = "\n".join([f"{k}: {v['score']}/10" for k,v in st.session_state.result.items()])
                try:
                    report = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": "你是一位资深心理学家，请基于以下人格分数，用温暖鼓励的语气写一段200字左右的性格解读，包括总体描述、优势、可成长空间和人际建议。"},
                                  {"role": "user", "content": f"人格分数：\n{summary}"}],
                        temperature=0.7, max_tokens=300
                    ).choices[0].message.content
                    st.markdown(report)
                except Exception as e:
                    st.error(f"生成报告失败: {str(e)}")
    with col2:
        if st.button("🔄 重新开始"):
            st.session_state.messages = []
            st.session_state.analysis_done = False
            st.session_state.result = None
            st.session_state.compare_list = []
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
