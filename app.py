# app.py
from dotenv import load_dotenv
load_dotenv()  # .env から OPENAI_API_KEY を読み込む

import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# --- ページ設定 ---
st.set_page_config(page_title="LLM専門家Webアプリ", page_icon="🤖", layout="centered")
st.title("🧙‍♂️ LLM専門家Webアプリ")
st.write("このアプリは、選択した専門家の視点からLLMが質問に回答するWebアプリケーションです。")
st.write("質問を入力し、専門家を選択して回答を得てください。")

# --- 専門家ロール定義（システムメッセージ） ---
EXPERT_SYSTEM_MESSAGES = {
    "ソムリエ": (
        "あなたは一流ソムリエです。日本語で親しみやすく、箇条書きを中心に回答してください。"
        "必ず『スタイル/銘柄例』『提供温度(℃)』『グラス』『ペアリング案』を含めてください。"
    ),
    "フィットネスコーチ": (
        "あなたは有資格のフィットネスコーチ兼栄養アドバイザーです。"
        "日本語で、実行手順を明確に、1日の食事例・運動メニュー・休息・注意点を具体的に提示してください。"
    ),
    "英会話チューター": (
        "You are a friendly English conversation tutor. Reply in simple English, "
        "add one JP note if helpful, and provide 2 alternative phrasings and 1 mini pronunciation tip."
    ),
}

# --- LLM 初期化 ---
def init_llm():
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ OPENAI_API_KEY が読み込めていません。.env または Secrets を確認してください。")
    # 課題向けに軽量・高品質モデルを既定に
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# --- 必須：関数（入力テキスト＆選択値→回答文字列） ---
def generate_response(user_text: str, role: str) -> str:
    system_msg = EXPERT_SYSTEM_MESSAGES.get(role, "")
    if not system_msg:
        return "ロールの設定が見つかりません。EXPERT_SYSTEM_MESSAGES を確認してください。"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", "{question}"),
        ]
    )
    chain = prompt | init_llm()
    try:
        result = chain.invoke({"question": user_text})
        return getattr(result, "content", str(result))
    except Exception as e:
        return f"エラーが発生しました: {e}"

# --- UI：ロール選択＋フォーム ---
role = st.radio("専門家ロールを選択：", list(EXPERT_SYSTEM_MESSAGES.keys()), horizontal=True)

with st.form("qa_form", clear_on_submit=False):
    user_text = st.text_area(
        "質問・相談内容：",
        placeholder="例）魚介に合うシャンパーニュと温度は？ / 体脂肪を落とす1週間メニュー / この英文を自然に言い換えて など",
        height=180,
    )
    submitted = st.form_submit_button("送信")

if submitted:
    if not user_text.strip():
        st.warning("テキストを入力してください。")
    else:
        with st.spinner("回答生成中…"):
            answer = generate_response(user_text, role)
        st.info(f"**選択ロール:** {role}")
        st.write("---")
        st.subheader("回答")
        st.write(answer)

st.caption("Powered by Streamlit × LangChain × OpenAI")
