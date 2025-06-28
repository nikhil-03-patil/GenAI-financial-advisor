# Phase 1+2+3 with Advice Saving & Clean Output
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import re
import seaborn as sns


# ---------------------------------------------
# Session Initialization
# ---------------------------------------------
if "saved_advice" not in st.session_state:
    st.session_state.saved_advice = []
if "latest_advice" not in st.session_state:
    st.session_state.latest_advice = ""

# ---------------------------------------------
# Load API Key
# ---------------------------------------------
load_dotenv()
genai_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=genai_api_key)

# Load Gemini model
model = genai.GenerativeModel("gemini-1.5-flash-latest")

from google.generativeai import GenerativeModel
PRIMARY_MODEL = "gemini-1.5-flash-latest"
FALLBACK_MODEL = "gemini-1.5-pro-latest"
primary_model = GenerativeModel(PRIMARY_MODEL)
fallback_model = GenerativeModel(FALLBACK_MODEL)


# ---------------------------------------------
# Generate with Fallback
# ---------------------------------------------
def generate_with_fallback(prompt: str) -> str:
    try:
        response = primary_model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "quota" in str(e).lower():
            st.warning("⚠️ Primary model quota reached. Switching to fallback (Gemini Pro)...")
            try:
                response = fallback_model.generate_content(prompt)
                return response.text
            except Exception as fallback_error:
                st.error("❌ Fallback model also failed.")
                st.code(str(fallback_error))
        else:
            st.error("❌ Gemini API Error:")
            st.code(str(e))
    return "⚠️ Unable to generate a response."

# ---------------------------------------------
# Text Cleaning Function
# ---------------------------------------------
def clean_gemini_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r'(?<=[a-zA-Z])(?=[0-9])', ' ', text)
    text = re.sub(r'(?<=[0-9])(?=[A-Z])', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

# ---------------------------------------------
# UI Setup
# ---------------------------------------------
st.set_page_config(page_title="GenAI Financial Health Advisor", layout="centered")
st.title("💬 GenAI Financial Health Advisor")

mode = st.selectbox("Choose what you want to do 👇", [
    "🔍 Ask a Financial Question",
    "📊 50/30/20 Budget Tool",
    "📈 Stock Market Analysis",
    "📘 Financial Planning Agent"

])

# ---------------------------------------------
# Mode 1: Q&A Prompt Mode
# ---------------------------------------------
if mode == "🔍 Ask a Financial Question":
    user_prompt = st.text_area("💡 Ask your financial question:", placeholder="E.g., Should I rent or buy a house in 2025?")

    if st.button("Get Advice"):
        if user_prompt.strip():
            with st.spinner("Gemini is thinking..."):
                try:
                    prompt = f"You are a helpful financial advisor. Be clear and practical.\n\nUser: {user_prompt}"
                    # response = model.generate_content(prompt)
                    response_text = generate_with_fallback(prompt)
                    advice_text = clean_gemini_text(response_text)

                    st.session_state.latest_advice = advice_text
                    st.success("🧠 AI Agent Advice:")
                    st.write(advice_text)
                except Exception as e:
                    st.error("❌ Gemini API Error:")
                    st.code(str(e))
        else:
            st.warning("Please enter a question.")

    if st.session_state.latest_advice:
        if st.button("💾 Save this advice", key="save_qna"):
            st.session_state.saved_advice.append(st.session_state.latest_advice)
            st.success("✅ Advice saved successfully.")

# ---------------------------------------------
# Mode 2: Budget Breakdown Tool
# ---------------------------------------------
elif mode == "📊 50/30/20 Budget Tool":
    st.subheader("📋 Your Monthly Budget Planner (50/30/20 Rule)")
    income = st.number_input("Enter your monthly income (Rs.)", min_value=1000, step=1000)

    if income > 0:
        needs = round(0.5 * income)
        wants = round(0.3 * income)
        savings = round(0.2 * income)

        # Text summary
        st.markdown("### 💡 Budget Breakdown")
        st.write(f"**Needs (50%)**: Rs. {needs}")
        st.write(f"**Wants (30%)**: Rs. {wants}")
        st.write(f"**Savings (20%)**: Rs. {savings}")

        # Data for charts
        import pandas as pd
        budget_df = pd.DataFrame({
            "Category": ["Needs", "Wants", "Savings"],
            "Amount": [needs, wants, savings]
        })

        # 🔹 Horizontal Bar Chart using Seaborn
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(x="Amount", y="Category", data=budget_df, palette="pastel", ax=ax)
        ax.set_title("Budget Allocation by Category")
        ax.set_xlabel("Amount (Rs.)")
        ax.set_ylabel("")
        st.pyplot(fig)

        # Gemini Commentary
        if st.button("Get AI Commentary"):
            with st.spinner("AI Agent is analyzing your budget..."):
                try:
                    budget_prompt = (
                        f"My income is Rs.{income}. Based on 50/30/20 rule: "
                        f"Rs.{needs} for needs, Rs.{wants} for wants, Rs.{savings} for savings. "
                        "Suggest how to optimize or improve this."
                    )
                    advice = model.generate_content(budget_prompt)
                    st.success("💬 AI Agent's Advice:")
                    st.write(advice.text)
                    st.session_state.latest_advice = advice.text
                except Exception as e:
                    st.error("❌ Gemini API error:")
                    st.code(str(e))

        # Save option
        if st.session_state.latest_advice:
            if st.button("💾 Save this advice", key="save_budget"):
                st.session_state.saved_advice.append(st.session_state.latest_advice)
                st.success("✅ Advice saved.")

# ---------------------------------------------
# Mode 3: Stock Market Analysis
# ---------------------------------------------
elif mode == "📈 Stock Market Analysis":
    import re
    import yfinance as yf

    def clean_gemini_text(text):
        text = text.replace("₹", "Rs. ")
        text = text.replace("–", "-").replace("—", "-")
        text = text.replace("“", '"').replace("”", '"').replace("’", "'")
        text = re.sub(r"\n{2,}", "\n\n", text)  # Keep paragraph spacing
        text = re.sub(r"[ ]{2,}", " ", text)
        return text.strip()

    stock_query = st.text_input("🔎 Ask about a stock (e.g., 'Analyze Tesla investment')")

    if st.button("Analyze Stock"):
        if stock_query.strip():
            with st.spinner("AI Agent is parsing your request..."):
                try:
                    # Step 1: Extract stock ticker using Gemini
                    ticker_prompt = (
                        f"You are an AI that extracts stock tickers from finance questions.\n"
                        f"Extract the correct stock ticker (e.g., TSLA, AAPL, GOOG) from this:\n"
                        f"'{stock_query}'\n"
                        "Only return the ticker (no explanation)."
                    )
                    ticker_response = model.generate_content(ticker_prompt)
                    ticker = ticker_response.text.strip().upper()

                    # Step 2: Get stock info
                    stock = yf.Ticker(ticker)
                    info = stock.info

                    company = info.get("longName", ticker)
                    price = info.get("currentPrice", "N/A")
                    pe = info.get("trailingPE", "N/A")
                    market_cap = info.get("marketCap", "N/A")
                    year_high = info.get("fiftyTwoWeekHigh", "N/A")
                    year_low = info.get("fiftyTwoWeekLow", "N/A")

                    # Step 3: Display data nicely
                    st.subheader(f"📄 Overview for {company} ({ticker})")
                    st.markdown(f"""
                    - **Current Price:** ${price}  
                    - **Market Cap:** {market_cap}  
                    - **P/E Ratio:** {pe}  
                    - **52-Week High:** ${year_high}  
                    - **52-Week Low:** ${year_low}
                    """)

                    # Step 4: Ask Gemini for investment overview
                    insights_prompt = (
                        f"Here is financial data for {company} ({ticker}):\n"
                        f"- Current Price: ${price}\n"
                        f"- Market Cap: {market_cap}\n"
                        f"- P/E Ratio: {pe}\n"
                        f"- 52-week High: ${year_high}\n"
                        f"- 52-week Low: ${year_low}\n\n"
                        "Now:\n"
                        "1. Give a short, investor-friendly summary of the stock.\n"
                        "2. Compare it briefly to other companies in its sector.\n"
                        "3. Categorize the investment potential as one of the following:\n"
                        "- High Potential\n"
                        "- Moderate Potential\n"
                        "- Low Potential\n"
                        "Give this rating in a separate last line as: '📊 Investment Potential: High/Moderate/Low'."
                    )
                    insights = model.generate_content(insights_prompt)
                    insight_text = clean_gemini_text(insights.text)

                    st.session_state.latest_advice = insight_text

                    # Step 5: Render final investment advice
                    st.success("📈 AI Investment Overview")
                    st.markdown(insight_text)

                except Exception as e:
                    st.error("❌ Failed to analyze stock:")
                    st.code(str(e))
        else:
            st.warning("Please enter a stock-related query.")

    if st.session_state.latest_advice:
        if st.button("💾 Save this advice", key="save_stock"):
            st.session_state.saved_advice.append(st.session_state.latest_advice)
            st.success("✅ Advice saved successfully.")






# -----------------------
# Mode 4: Personalized Financial Plan Agent
# -----------------------
elif mode == "📘 Financial Planning Agent":
    from fpdf import FPDF
    import base64
    import textwrap
    from datetime import datetime

    def sanitize_text(text):
        """Replace problematic Unicode characters not supported in latin-1"""
        return (
            text.replace("₹", "Rs. ")
                .replace("–", "-")
                .replace("—", "-")
                .replace("“", '"')
                .replace("”", '"')
                .replace("’", "'")
                .replace("\n", " ")
                .strip()
        )

    class StyledPDF(FPDF):
        def header(self):
            self.set_font("Helvetica", 'B', 16)
            self.set_text_color(30, 30, 30)
            self.cell(0, 10, "Financial Planning Report", ln=True, align="C")
            self.set_font("Helvetica", size=10)
            self.set_text_color(100)
            self.cell(0, 8, f"Generated on: {datetime.now().strftime('%d %b %Y')}", ln=True, align="C")
            self.ln(4)

        def section_title(self, title):
            self.set_font("Helvetica", 'B', 13)
            self.set_text_color(50, 50, 50)
            self.cell(0, 10, title, ln=True)
            self.set_text_color(0)
            self.set_font("Helvetica", size=11)

        def add_wrapped_paragraph(self, text):
            self.set_font("Helvetica", size=11)
            self.set_text_color(0)
            wrapped = textwrap.wrap(text, width=90)
            for line in wrapped:
                self.cell(0, 8, line, ln=True)
            self.ln(4)

        def add_key_value_block(self, data_dict):
            self.set_font("Helvetica", size=11)
            for key, value in data_dict.items():
                self.set_font("Helvetica", 'B', 11)
                self.cell(50, 8, f"{key}:", ln=False)
                self.set_font("Helvetica", size=11)
                self.cell(0, 8, str(value), ln=True)
            self.ln(4)

    def create_pdf_with_layout(user_inputs, plan_text, filename="financial_plan.pdf"):
        plan_text = sanitize_text(plan_text)

        pdf = StyledPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Section 1: User Inputs
        pdf.section_title("User Profile")
        pdf.add_key_value_block(user_inputs)

        # Section 2: Generated Financial Plan
        pdf.section_title("AI-Generated Financial Plan")
        pdf.add_wrapped_paragraph(plan_text)

        pdf.output(filename)

        with open(filename, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📄 Download PDF</a>'
        return href

    # --- UI Form ---
    st.subheader("🧾 Tell us about yourself:")

    with st.form("plan_form"):
        income = st.number_input("Monthly Income (Rs.)", min_value=1000, step=1000)
        family_size = st.slider("Family Size (including you)", 1, 10, 2)
        age = st.selectbox("Your Age Range", ["18-25", "26-35", "36-45", "46-60", "60+"])
        goal = st.text_area("Your Primary Financial Goal", placeholder="E.g., Buy a house in 5 years")
        existing_debt = st.number_input("Monthly Loan/EMI Payments (Rs.)", min_value=0, step=1000)
        risk = st.selectbox("Your Risk Appetite", ["Low", "Medium", "High"])

        submitted = st.form_submit_button("🧠 Generate My Plan")

    # --- AI Response ---
    if submitted:
        with st.spinner("Generating your personalized financial plan..."):
            try:
                plan_prompt = f"""
You are a smart financial planning assistant.

The user provides:
- Income: Rs.{income} / month
- Family size: {family_size}
- Age group: {age}
- Goal: {goal}
- Debt: Rs.{existing_debt}/month
- Risk appetite: {risk}

Based on this, generate a complete financial plan. Include:
1. Budget allocation
2. Recommended savings target
3. Insurance needs
4. Investment strategy
5. Clear next steps

Respond in friendly and readable language.
                """

                response = model.generate_content(plan_prompt)
                plan_text = sanitize_text(response.text)
                st.success("📋 Your Personalized Financial Plan:")
                st.write(plan_text)

                st.session_state.latest_advice = plan_text

                # Save
                if st.button("💾 Save this plan", key="save_plan"):
                    st.session_state.saved_advice.append(plan_text)
                    st.success("✅ Plan saved to your advice history.")

                # PDF Download
                user_data = {
                    "Monthly Income": f"Rs. {income}",
                    "Family Size": family_size,
                    "Age Group": age,
                    "Financial Goal": goal,
                    "Existing Debt": f"Rs. {existing_debt}",
                    "Risk Appetite": risk
                }

                st.markdown(create_pdf_with_layout(user_data, plan_text), unsafe_allow_html=True)

            except Exception as e:
                st.error("❌ Gemini API error:")
                st.code(str(e))


# ---------------------------------------------
# Saved Advice Viewer
# ---------------------------------------------
st.markdown("---")
with st.expander("📂 View Saved Advice"):
    if st.session_state.saved_advice:
        for i, advice in enumerate(st.session_state.saved_advice, 1):
            st.markdown(f"**📝 Advice {i}:**")
            st.write(advice)
            st.markdown("---")
    else:
        st.info("No saved advice yet. Use 💾 after any analysis to store it.")

