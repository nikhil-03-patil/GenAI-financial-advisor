
 🧠 GenAI Financial Health Advisor

An Agentic AI-powered application that provides personalized financial guidance using Google Gemini, financial logic, and real-time market data. This AI agent simulates the capabilities of a professional financial advisor — capable of answering user-specific queries, analyzing budgets, evaluating investment opportunities, and generating full financial plans.

---


> 🔗 Try the App: [https://master-nikhilaiagent.streamlit.app](https://master-nikhilaiagent.streamlit.app)



---

 🚀 What Does It Do?

#This application solves real-world BFSI pain points by helping users:
- Understand how to budget using the 50/30/20 rule.
- Ask personalized financial questions and receive human-like, reasoned responses.
- Analyze the financial standing of any stock using live data.
- Generate a downloadable, structured financial plan — tailored to income, goals, risk, and more.

---

 🤖 Why Agentic AI?

This solution is designed with an Agentic AI paradigm, meaning:
- 🧠 It autonomously reasons through goals based on user prompts.
- 🧰 It uses external tools (e.g., yFinance, plotting libraries) to gather or visualize information.
- 📋 It completes tasks end-to-end, such as generating personalized PDF financial plans.
- 💡 It is modular and extensible — each module is task-specific and goal-oriented.

---

 🛠 How It Works (Tech Overview)

| Module                          | Functionality                                                                 |
|---------------------------------|------------------------------------------------------------------------------|
| 🔍 Prompt Q&A (LLM Agent)        | Accepts free-form questions. Gemini generates clear, context-aware answers. |
| 📊 Budget Breakdown Tool         | Uses income to apply the 50/30/20 rule, generates pie + bar charts, adds AI insights. |
| 📈 Stock Market Analysis         | Extracts stock ticker, pulls live metrics (P/E, market cap, etc.) from yFinance, and prompts Gemini to evaluate investment quality. |
| 📘 Financial Planning Agent      | Takes detailed user profile inputs, generates a custom plan, and exports it as a downloadable PDF using FPDF. |

---

 🔧 Tech Stack

- Frontend/UI: Streamlit
- LLM: Google Gemini 1.5 Flash (via `google-generativeai`)
- Visualization: Matplotlib, Seaborn, Plotly
- Finance Data: yFinance (stock data)
- PDF Reporting: FPDF
- Environment Management: python-dotenv
- Deployment:** GitHub + Streamlit Cloud

---

 📁 Folder Structure

```
financial-health-advisor/
├── app.py                  # Main Streamlit app
├── .env.example            # Sample for your Gemini API key
├── requirements.txt        # Dependencies
├── assets/                 # Diagrams or logos (optional)
├── saved_advice/           # (Future scope) Local saved output
└── .gitignore              # Ignores .env, venv, __pycache__, etc.
```

---

 🔐 Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/nikhil-03-patil/GenAI-financial-advisor.git
   cd GenAI-financial-advisor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Gemini API key:
   ```env
   GEMINI_API_KEY=your_google_gemini_key_here
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

---

 🌐 Deploying to Streamlit Cloud

1. Push the project to GitHub.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud).
3. Connect your repo and deploy `app.py`.
4. Add the `GEMINI_API_KEY` in the Secrets tab.
5. 🎉 Your app is live!

---

 🧪 Example Prompts to Try

- `"What should I do with ₹70k/month income and ₹40k expenses?"`
- `"How should I plan for my child's education in 10 years?"`
- `"Analyze investment in Apple stock"`
- `"Create a financial plan. I earn ₹50k/month, have a loan, and want to buy a house."`

---

 👥 Team & Contribution

Developed by Nikhil Patil for the DSW GenAI Hackathon, with a focus on modular, deployable, real-world Agentic AI design.

---

 📄 License

Licensed under the Apache 2.0 License.
