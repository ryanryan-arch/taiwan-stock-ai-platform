from pathlib import Path

import streamlit as st


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

STYLE_PATH = (
    PROJECT_ROOT
    / "assets"
    / "style.css"
)


def load_global_css():
    """
    載入全站共用 CSS。

    如果 CSS 檔案不存在，只顯示提醒，
    不讓整個網站停止執行。
    """

    if not STYLE_PATH.exists():
        st.warning(
            "找不到 assets/style.css，"
            "網站將使用 Streamlit 預設樣式。"
        )
        return

    css_content = STYLE_PATH.read_text(
        encoding="utf-8"
    )

    st.markdown(
        f"<style>{css_content}</style>",
        unsafe_allow_html=True,
    )


def render_sidebar_info():
    """
    顯示側邊欄的系統摘要。
    """

    st.sidebar.divider()

    st.sidebar.markdown(
        """
        ### 系統資訊

        **模型**：XGBoost  
        **股票數量**：85 檔  
        **產業族群**：17 個  
        **模型特徵**：32 個  
        **預測期間**：未來 5 個交易日  
        **更新方式**：交易日晚間自動更新
        """
    )


def render_footer():
    """
    顯示共用頁尾。
    """

    st.markdown(
        """
        <div class="ai-footer">
            台股 AI 智慧選股與五日趨勢預測平台
            ｜僅供課程研究與模型驗證
        </div>
        """,
        unsafe_allow_html=True,
    )