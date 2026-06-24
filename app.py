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

# ---------- 持久化存储 ----------
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

# ---------- 中文/多语言字体智能回退 ----------
def get_chinese_font():
    font_names = ['PingFang SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    for font in font_names:
        try:
            if fm.findfont(font):
                return font
        except:
            continue
    return 'sans-serif'

plt.rcParams['font.sans-serif'] = [get_chinese_font()]
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

# ---------- 读取 DeepSeek 配置 ----------
if "DEEPSEEK_API_KEY" not in st.secrets:
    st.error("请在 Streamlit Cloud 的 Settings → Secrets 中添加 DEEPSEEK_API_KEY")
    st.stop()

client = OpenAI(
    api_key=st.secrets["DEEPSEEK_API_KEY"],
    base_url=st.secrets.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)
model = st.secrets.get("DEEPSEEK_MODEL", "deepseek-chat")

# ---------- 核心 API 与分析逻辑 ----------
def analyze_personality(text):
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
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if not json_match:
            st.error(f"❌ 大模型返回格式异常，未找到JSON。返回原始内容：\n{result}")
            return None
        return json.loads(json_match.group())
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON解析失败，请检查大模型输出格式。错误详情：{str(e)}\n原始内容片段：{result[:200]}")
        return None
    except Exception as e:
        st.error(f"❌ API调用发生致命错误：{str(e)}")
        return None

def summarize_text(text):
    if len(text) < 1000: return text
    try:
        prompt = f"将以下心理对话文本，压缩摘要成200字以内的核心情感与内容摘要，保留用户表达的真实情绪，不要丢失细节：\n\n{text}"
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5, max_tokens=300
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return text

# ---------- 心理学图谱转换 (大五人格 -> MBTI) ----------
def compute_mbti_from_big5(big5):
    """
    通用心理学近似映射关系：
    外向型(E) ← 社交外向
    直觉型(N) ← 开放心态
    情感型(F) ← 同理合作
    判断型(J) ← 认真负责
    """
    mbti_map = {}
    scores = {
        "E": big5["社交外向"]["score"],
        "N": big5["开放心态"]["score"],
        "F": big5["同理合作"]["score"],
        "J": big5["认真负责"]["score"]
    }
    
    # 计算每一对的中值
    mbti_letters = []
    for trait, score in scores.items():
        mbti_map[trait] = score
        # 反向字母得分：I = 10 - E, S = 10 - N 等
        opposite_trait = {"E": "I", "N": "S", "F": "T", "J": "P"}[trait]
        mbti_map[opposite_trait] = 10 - score
        
        # 判定字母
        mbti_letters.append(trait if score >= 5 else opposite_trait)
    
    mbti_type = "".join(mbti_letters)
    return mbti_type, mbti_map

# ---------- 全新绘图表 (改为了 MBTI 四极雷达 + 大五条形图) ----------
def draw_mbti_radar(mbti_map, mbti_type):
    """绘制 MBTI 倾向性 4极雷达图 (代替原5角雷达)"""
    # 轴名顺序：E-I, N-S, F-T, J-P
    axes = ['E/I', 'N/S', 'F/T', 'J/P']
    # 取对应两端的分数，一端是正分，另一端是反分
    # 为了绘制四极，我们分别取 E,I,N,S,F,T,J,P 投射到0-10
    # 这里我们将 4 个轴分别画成 0-10 的 4 条线，0在中心，10在外围
    # 为了让图形有意义，直接绘制 4 条线。
    # 现在改为画一个 4象限雷达图（4 条轴伸出去）
    # 每条轴的两端分别代表两个字母。例如 E(0-10) 和 I(0-10)。
    
    # 为了标准化四极显示，取 4 条轴，评分 0-10。0 代表一边，10 代表另一边。
    # 这里我们采用“两极单轴”法，即 E/I 共用一根 0-10 的轴，0 代表 I，10 代表 E。
    # 所以值 = 0.5 * (E - I) + 5. 如果 E = 8, I=2, 得分 = 5 + 3 = 8 (偏向E)
    dims = {
        "E/I": 5 + (mbti_map["E"] - mbti_map["I"]) / 2,
        "N/S": 5 + (mbti_map["N"] - mbti_map["S"]) / 2,
        "F/T": 5 + (mbti_map["F"] - mbti_map["T"]) / 2,
        "J/P": 5 + (mbti_map["J"] - mbti_map["P"]) / 2
    }
    
    labels = list(dims.keys())
    values = [dims[k] for k in labels]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#f5f5f7')
    ax.set_facecolor('#fafafa')

    # 颜色改为 MBTI 对应的科技渐变紫/蓝
    color = '#8B5CF6'  # 紫色
    ax.fill(angles, values, color=color, alpha=0.25)
    ax.plot(angles, values, color=color, linewidth=3.5, marker='D', markersize=8,
            markerfacecolor='white', markeredgecolor=color, markeredgewidth=2.5)

    ax.set_ylim(0, 10)
    ax.set_rticks([2, 5, 8])
    ax.set_rlabel_position(45)
    ax.grid(True, linestyle='--', color='#d1d1d6', linewidth=1, alpha=0.8)
    ax.spines['polar'].set_visible(False)

    # 在极坐标绘图时，注意文字位置，并支持英文替换
    current_font = plt.rcParams['font.sans-serif'][0]
    for label, angle in zip(labels, angles[:-1]):
        rotation = np.rad2deg(angle)
        if 90 <= rotation <= 270: ha = 'right'
        else: ha = 'left'
        if 0 <= rotation <= 180: va = 'bottom'
        else: va = 'top'
        
        if current_font == 'sans-serif':
            # 如果无中文字体，自动翻译为英文
            en_label = {"E/I": "Extraversion", "N/S": "Intuition", "F/T": "Feeling", "J/P": "Judging"}
            display_label = en_label.get(label, label)
        else:
            display_label = label

        ax.text(angle, 10.8, display_label, ha=ha, va=va, 
                fontsize=12, fontweight='600', color='#4C1D95')

    # 中间显示 MBTI 类型
    ax.text(0, 0, mbti_type, fontsize=28, fontweight='900', color='#5B21B6',
            ha='center', va='center')
    return fig

def draw_big5_bar(big5):
    """绘制大五人格的水平条形图（替换掉原来的雷达图）"""
    labels = list(big5.keys())
    values = [big5[k]["score"] for k in labels]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#f5f5f7')
    ax.set_facecolor('#f8f8fa')
    
    # 绘制黑色底、霓虹蓝的酷炫条形
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, height=0.6, color='#0071e3', edgecolor='white', linewidth=1.5)
    
    ax.set_yticks(y_pos)
    current_font = plt.rcParams['font.sans-serif'][0]
    if current_font == 'sans-serif':
        en_labels = {"开放心态": "Openness", "认真负责": "Conscientiousness", "社交外向": "Extraversion", 
                     "同理合作": "Agreeableness", "情绪稳定": "Emotional Stability"}
        display_labels = [en_labels.get(l, l) for l in labels]
    else:
        display_labels = labels

    ax.set_yticklabels(display_labels, fontsize=11, fontweight='500')
    ax.set_xlim(0, 10)
    ax.set_xticks([2,4,6,8,10])
    ax.set_xticklabels(['2','4','6','8','10'], color='#888')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#DDD')
    ax.spines['left'].set_color('#DDD')
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # 在条形末尾添加数值
    for bar, val in zip(bars, values):
        ax.text(val + 0.2, bar.get_y() + bar.get_height()/2, 
                f"{val}", va='center', ha='left', fontsize=12, fontweight='bold', color='#1d1d1f')

    plt.tight_layout()
    return fig

def generate_card(big5, mbti_type, mbti_map):
    """生成分享卡片：组合了新的雷达图和条形图"""
    fig = plt.figure(figsize=(10, 5))
    fig.patch.set_facecolor('#f5f5f7')
    
    # 左侧画 MBTI 雷达图
    ax1 = fig.add_subplot(1, 2, 1, projection='polar')
    # 重新构建 MBTI 绘图逻辑以适配单个子图
    dims = {
        "E/I": 5 + (mbti_map["E"] - mbti_map["I"]) / 2,
        "N/S": 5 + (mbti_map["N"] - mbti_map["S"]) / 2,
        "F/T": 5 + (mbti_map["F"] - mbti_map["T"]) / 2,
        "J/P": 5 + (mbti_map["J"] - mbti_map["P"]) / 2
    }
    labels = list(dims.keys())
    values = [dims[k] for k in labels] + [dims[labels[0]]]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]
    
    ax1.fill(angles, values, color='#8B5CF6', alpha=0.25)
    ax1.plot(angles, values, color='#8B5CF6', linewidth=3, marker='D')
    ax1.set_ylim(0,10)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.spines['polar'].set_visible(False)
    ax1.text(0, 0, mbti_type, fontsize=24, fontweight='900', color='#5B21B6', ha='center', va='center')
    
    # 右侧画信息摘要
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')
    traits = [f"{k}: {v['score']}/10" for k, v in big5.items()]
    summary = "\n".join(traits)
    ax2.text(0.1, 0.8, "🧠 文以辨心", fontsize=18, fontweight='bold', color='#1d1d1f')
    ax2.text(0.1, 0.6, f"MBTI 类型：{mbti_type}", fontsize=14, fontweight='600', color='#4C1D95')
    ax2.text(0.1, 0.4, summary, fontsize=12, color='#555', linespacing=2)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    ax2.text(0.1, 0.1, f"分析时间：{now}", fontsize=10, color='#999')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    return buf

# ---------- 会话状态 ----------
if "messages" not in st.session_state: st.session_state.messages = []
if "analysis_done" not in st.session_state: st.session_state.analysis_done = False
if "result" not in st.session_state: st.session_state.result = None
if "history" not in st.session_state: st.session_state.history = load_history()
if "selected_mode" not in st.session_state: st.session_state.selected_mode = "💬 对话引导"
if "compare_list" not in st.session_state: st.session_state.compare_list = []

# ---------- 页面主卡片 ----------
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="title-text">🧠 文以辨心</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">AI · 心理学 — 大五人格 + MBTI 轮廓透析</div>', unsafe_allow_html=True)

mode = st.radio("选择模式", ["💬 对话引导", "⚡ 直接分析", "📊 历史对比", "🖼️ 分享卡片"], horizontal=True, key="mode_radio")
st.session_state.selected_mode = mode

if mode == "💬 对话引导":
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        elif msg.get("temp"):
            st.markdown(f'<div class="typing-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble ai-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.session_state.messages.append({"role": "ai", "content": "嗨！我想更了解你。随便说点什么吧，比如最近的心情、让你开心或烦恼的事…"})
        st.rerun()

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("输入消息", placeholder="在这里打字… 按 Enter 发送", label_visibility="collapsed")
        st.markdown('<div class="send-hint">↵ 按回车发送消息</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1: send_btn = st.form_submit_button("发送")
        with col2: analyze_chat_btn = st.form_submit_button("🔍 分析对话")

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
        if len(full_text) < 20: st.warning("对话内容太少，再多聊两句吧～")
        else:
            with st.spinner("AI 正在分析你们的对话（长对话会自动压缩以节省Token）…"):
                analysis_text = summarize_text(full_text)
                data = analyze_personality(analysis_text)
            if data:
                st.session_state.result = data
                st.session_state.analysis_done = True
                record = {"time": datetime.now().strftime("%m-%d %H:%M"), "text": full_text[:60]+"...", "result": data}
                st.session_state.history.insert(0, record)
                if len(st.session_state.history) > 5: st.session_state.history.pop()
                save_history(st.session_state.history)
                st.rerun()

elif mode == "⚡ 直接分析":
    direct_text = st.text_area("写下你此刻的想法、心情或任何一段文字", height=160)
    if st.button("🔍 开始分析"):
        if len(direct_text) < 10: st.warning("至少输入10个字，才能获得准确分析")
        else:
            with st.spinner("AI 正在解读…"):
                data = analyze_personality(direct_text)
            if data:
                st.session_state.result = data
                st.session_state.analysis_done = True
                record = {"time": datetime.now().strftime("%m-%d %H:%M"), "text": direct_text[:60]+"...", "result": data}
                st.session_state.history.insert(0, record)
                if len(st.session_state.history) > 5: st.session_state.history.pop()
                save_history(st.session_state.history)
                st.rerun()

elif mode == "📊 历史对比":
    if not st.session_state.history: st.info("暂无历史记录，先去分析吧～")
    else:
        if st.button("🗑️ 清空所有历史记录"):
            st.session_state.history = []
            if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
            st.rerun()
        st.markdown("### 📜 最近五次分析")
        for i, item in enumerate(st.session_state.history):
            col1, col2 = st.columns([4,1])
            with col1: st.write(f"**{item['time']}**：{item['text']}")
            with col2:
                if st.button("选择", key=f"sel_{i}"):
                    if len(st.session_state.compare_list) < 2: st.session_state.compare_list.append(item)
                    else: st.session_state.compare_list = [st.session_state.compare_list[-1], item]
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
                    values = [scores[k]["score"] for k in all_labels] + [scores[all_labels[0]]["score"]]
                    ax.fill(angles, values, color=colors[idx], alpha=0.1)
                    ax.plot(angles, values, color=colors[idx], linewidth=2.5, marker='o')
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(all_labels, fontsize=10)
                ax.set_ylim(0,10)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                ax.grid(color='#d1d1d6', linestyle='--', linewidth=0.6)
                st.pyplot(fig)
            except Exception as e: st.error(f"对比图生成出错: {str(e)}")
            if st.button("清除对比"): st.session_state.compare_list = []; st.rerun()

elif mode == "🖼️ 分享卡片":
    if st.session_state.result:
        if st.button("📸 生成分享卡片"):
            mbti_type, mbti_map = compute_mbti_from_big5(st.session_state.result)
            buf = generate_card(st.session_state.result, mbti_type, mbti_map)
            st.download_button("⬇️ 下载 PNG", data=buf, file_name="personality_card.png", mime="image/png")
    else:
        st.warning("请先完成一次人格分析")

# ---------- 结果展示区：大五人格横向条 + MBTI 四极图 ----------
if st.session_state.analysis_done and st.session_state.result:
    st.markdown("---")
    st.success("分析完成！")
    
    # 1. 计算 MBTI
    mbti_type, mbti_map = compute_mbti_from_big5(st.session_state.result)
    st.subheader(f"🧬 你的 MBTI 倾向：**{mbti_type}**")
    
    # 2. 显示双图表 (新的组合：左侧 MBTI 雷达图，右侧 大五人格条形图)
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.caption("⚡ 四维倾向极坐标 (MBTI)")
        fig_mbti = draw_mbti_radar(mbti_map, mbti_type)
        st.pyplot(fig_mbti)
    
    with col2:
        st.caption("📊 大五人格维度 (条形图)")
        fig_bar = draw_big5_bar(st.session_state.result)
        st.pyplot(fig_bar)

    st.markdown("### 🧠 深度维度剖析")
    for trait, info in st.session_state.result.items():
        label = "高" if info['score']>=7 else "中" if info['score']>=4 else "低"
        with st.expander(f"{trait}  {info['score']}/10 ({label})"):
            st.write(f"**AI 原文依据**：{info['reason']}")

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
                except Exception as e: st.error(f"生成报告失败: {str(e)}")
    with col2:
        if st.button("🔄 重新开始"):
            st.session_state.messages = []; st.session_state.analysis_done = False; st.session_state.result = None; st.session_state.compare_list = []; st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
