"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
from dotenv import load_dotenv
import streamlit as st
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain import SerpAPIWrapper
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
import utils
import constants as ct



############################################################
# 設定関連
############################################################
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # Agent Executorを作成
    initialize_agent_executor()


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []
        # 会話履歴の合計トークン数を加算する用の変数
        st.session_state.total_tokens = 0

        # フィードバックボタンで「はい」を押下した後にThanksメッセージを表示するためのフラグ
        st.session_state.feedback_yes_flg = False
        # フィードバックボタンで「いいえ」を押下した後に入力エリアを表示するためのフラグ
        st.session_state.feedback_no_flg = False
        # LLMによる回答生成後、フィードバックボタンを表示するためのフラグ
        st.session_state.answer_flg = False
        # フィードバックボタンで「いいえ」を押下後、フィードバックを送信するための入力エリアからの入力を受け付ける変数
        st.session_state.dissatisfied_reason = ""
        # フィードバック送信後にThanksメッセージを表示するためのフラグ
        st.session_state.feedback_no_reason_send_flg = False


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


def initialize_logger():
    """
    ログ出力の設定
    """
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)

    logger = logging.getLogger(ct.LOGGER_NAME)

    if logger.hasHandlers():
        return

    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )
    log_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)


def initialize_agent_executor():
    """
    画面読み込み時にAgent Executor（AIエージェント機能の実行を担当するオブジェクト）を作成
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにAgent Executorが作成済みの場合、後続の処理を中断
    if "agent_executor" in st.session_state:
        return
    
    # 消費トークン数カウント用のオブジェクトを用意
    st.session_state.enc = tiktoken.get_encoding(ct.ENCODING_KIND)
    
    st.session_state.llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE, streaming=True)

    # 各Tool用のChainを作成
    st.session_state.customer_doc_chain = utils.create_rag_chain(ct.DB_CUSTOMER_PATH)
    st.session_state.service_doc_chain = utils.create_rag_chain(ct.DB_SERVICE_PATH)
    st.session_state.company_doc_chain = utils.create_rag_chain(ct.DB_COMPANY_PATH)
    st.session_state.rag_chain = utils.create_rag_chain(ct.DB_ALL_PATH)

    # Web検索用のToolを設定するためのオブジェクトを用意
    search = SerpAPIWrapper()
    # Agent Executorに渡すTool一覧を用意
    tools = [
        # 会社に関するデータ検索用のTool
        Tool(
            name=ct.SEARCH_COMPANY_INFO_TOOL_NAME,
            func=utils.run_company_doc_chain,
            description=(
                "会社に関する情報を検索します。このToolは、"
                "会社の概要、歴史、所在地、連絡先情報などを取得するために使用します。"
                "入力には具体的な会社名を含めてください。"
            )
        ),
        # サービスに関するデータ検索用のTool
        Tool(
            name=ct.SEARCH_SERVICE_INFO_TOOL_NAME,
            func=utils.run_service_doc_chain,
            description=(
                "サービスに関する情報を検索します。このToolは、"
                "提供されているサービスの詳細、料金、利用条件などを取得するために使用します。"
                "入力にはサービス名またはカテゴリを含めてください。"
            )
        ),
        # 顧客とのやり取りに関するデータ検索用のTool
        Tool(
            name=ct.SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_NAME,
            func=utils.run_customer_doc_chain,
            description=(
                "顧客とのやり取りに関する情報を検索します。このToolは、"
                "過去の問い合わせ履歴、対応状況、顧客満足度などを取得するために使用します。"
                "入力には顧客名または問い合わせIDを含めてください。"
            )
        ),
        # Web検索用のTool
        Tool(
            name=ct.SEARCH_WEB_INFO_TOOL_NAME,
            func=search.run,
            description=(
                "インターネット上の情報を検索します。このToolは、"
                "一般的な質問や外部情報を取得するために使用します。"
                "入力には具体的な検索クエリを含めてください。"
            )
        ),
        # 全データを横断的に検索するTool
        Tool(
            name=ct.SEARCH_ALL_DATA_TOOL_NAME,
            func=utils.run_all_data_doc_chain,
            description=(
                "すべてのデータベースを横断的に検索します。このToolは、"
                "会社、サービス、顧客情報を含む複数のデータソースから情報を取得するために使用します。"
                "特定のカテゴリに限定されない質問や、複数のカテゴリにまたがる質問に適しています。"
                "例えば、『EcoTeeに関するすべての情報を教えてください』や、"
                "『顧客とサービスに関する関連情報をまとめて教えてください』といった質問に対応します。"
                "入力には具体的なキーワードを含めてください。"
            )
        ),
    ]

    # Agent Executorの作成
    st.session_state.agent_executor = initialize_agent(
        llm=st.session_state.llm,
        tools=tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=ct.AI_AGENT_MAX_ITERATIONS,
        early_stopping_method="generate",
        handle_parsing_errors=True
    )