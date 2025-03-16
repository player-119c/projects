from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from apikey import GROQ_KEY

llm = ChatGroq(api_key=GROQ_KEY, model="deepseek-r1-distill-llama-70b", temperature=0.5)

# Market data analyst chain
market_prompt = PromptTemplate(
    input_variables=["ticker"],
    template=(
        "For {ticker}, provide detailed and up-to-date financial data. Include current stock price, "
        "volume, key financial ratios (e.g., P/E, P/B, dividend yield), recent price trends, and relevant market indicators."
    )
)
market_data_chain = market_prompt | llm

# Sentiment analysis chain
senti_prompt = PromptTemplate(
    input_variables=['ticker'],
    template=(
        "For {ticker}, analyze recent news articles, social media posts, and expert commentary. "
        "Summarize the prevailing sentiment, highlight any key events, and note emerging trends that may impact the stock."
    )
)
sentiment_chain = senti_prompt | llm

# Macroeconomic analysis chain
macro_prompt = PromptTemplate(
    input_variables=['ticker'],
    template=(
        "For {ticker}, analyze the current macro-economic environment. "
        "Include key indicators such as GDP growth, inflation rates, interest rates, unemployment trends, "
        "and central bank policies. Summarize how these factors could impact the overall market and the asset."
    )
)
macro_analysis_chain = macro_prompt | llm

# Quantitative strategy chain
quant_prompt = PromptTemplate(
    input_variables=['market_data', 'sentiment_analysis', 'macro_analysis'],
    template=(
        "Using the detailed market data:\n{market_data}\n"
        "the sentiment analysis:\n{sentiment_analysis}\n"
        "and the macro-economic analysis:\n{macro_analysis}\n"
        "develop a sophisticated trading strategy. Outline a clear asset allocation, specify entry and exit points, "
        "detail risk management measures, and provide estimated expected returns. If applicable, incorporate algorithmic signals."
    )
)
strategy_chain = quant_prompt | llm

# Risk management chain
risk_prompt = PromptTemplate(
    input_variables=['strategy'],
    template=(
        "Evaluate the following trading strategy:\n{strategy}\n"
        "Identify potential risks such as market volatility, liquidity issues, or unexpected market events. "
        "Summarize your risk assessment in 4 concise bullet points, and state in the final bullet point whether the strategy meets an acceptable risk tolerance."
    )
)
risk_chain = risk_prompt | llm

# LCEL Pipeline
analysis_chain = (
    RunnableParallel(
        market_data=market_data_chain,
        sentiment_analysis=sentiment_chain,
        macro_analysis=macro_analysis_chain
    )
    | RunnablePassthrough.assign(strategy=strategy_chain)
    | RunnablePassthrough.assign(risk_assessment=risk_chain)
)

# Updated execution function
def run_ai_hedge_fund(ticker: str) -> None:
    result = analysis_chain.invoke({"ticker": ticker})
    filename=f"{ticker}_analysis.txt"
    
   # Write results to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"======== AI Hedge Fund Analysis for {ticker} ========\n\n")
        
        f.write("📈 Market Data Retrieved:\n")
        f.write("--------------------------------------------------\n")
        f.write(f"{result['market_data'].content}\n\n")
        
        f.write("📰 Market Sentiment Analysis:\n")
        f.write("--------------------------------------------------\n")
        f.write(f"{result['sentiment_analysis'].content}\n\n")
        
        f.write("🌐 Macro-Economic Analysis:\n")
        f.write("--------------------------------------------------\n")
        f.write(f"{result['macro_analysis'].content}\n\n")
        
        f.write("📊 Developed Trading Strategy:\n")
        f.write("--------------------------------------------------\n")
        f.write(f"{result['strategy'].content}\n\n")
        
        f.write("⚠️ Risk Assessment:\n")
        f.write("--------------------------------------------------\n")
        f.write(f"{result['risk_assessment'].content}\n\n")
        
        f.write("==============================================\n")

# Execute analysis
run_ai_hedge_fund("MSFT")
