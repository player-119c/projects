from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from apikey import GROQ_KEY, ALPHAVANTAGE_API_KEY, NEWS_API_KEY_4
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
from newsapi.newsapi_client import NewsApiClient
import yfinance as yf
import re

# --- Output Cleaning ---

def remove_think_tags(filename):
    with open(filename, 'r') as file:
        content = file.read()
    
    cleaned_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    with open(f'cleaned_{filename}', 'w') as file:
        file.write(cleaned_content)


        

# --- LLM Initialization ---
llm = ChatGroq(model='deepseek-r1-distill-qwen-32b', api_key=GROQ_KEY, temperature=0.7)
newsapi = NewsApiClient(api_key=NEWS_API_KEY_4)

# --- Data Collection ---
def get_market_data(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    return {
        "price": hist['Close'].iloc[-1],
        "volume": hist['Volume'].iloc[-1],
        "pe_ratio": stock.info.get('trailingPE', 'N/A'),
        "pb_ratio": stock.info.get('priceToBook', 'N/A'),
        "dividend_yield": stock.info.get('dividendYield', 'N/A'),
        "beta": stock.info.get('beta', 'N/A')
    }

def get_news(ticker: str) -> list:
    return newsapi.get_everything(q=f"{ticker} stock", language='en', sort_by='relevancy', page_size=5)['articles']

def analyze_sentiment(articles: list) -> str:
    news_text = "\n".join([f"{a['title']}: {a['description']}" for a in articles])
    return llm.invoke(f"Analyze market sentiment from:\n{news_text}").content

def analyze_macro() -> str:
    news = newsapi.get_top_headlines(q="economy   GDP  inflation  interest rates  unemployment", category='business', page_size=5)['articles']
    news_text = "\n".join([f"{a['title']}: {a['description']}" for a in news])
    return llm.invoke(f"Analyze macroeconomic trends from:\n{news_text}").content

# --- Agent Strategies ---


def stan_druckenmiller_strategy(market_data, sentiment, macro):
    prompt = f"""
    Stan Druckenmiller's Macro Trading Strategy:
    - Trade based on macroeconomic trends.
    - Interest rates, inflation, and central bank policy are key.
    Market Data: {market_data}
    Macro Trends: {macro}

    Decision:
    - Buy if macro tailwinds support growth
    - Sell if macro conditions turn bearish
    """
    return llm.invoke(prompt).content

def warren_buffett_strategy(market_data, sentiment, macro):
    prompt = f"""
    Warren Buffett's Long-Term Value Investing Strategy:
    - Buy and hold strong businesses.
    - Ignore market noise.
    Market Data: {market_data}
    Macro Trends: {macro}

    Decision:
    - Buy if company has strong brand and cash flow
    - Sell if fundamentals decline
    """
    return llm.invoke(prompt).content



# --- Risk Manager ---
risk_prompt = PromptTemplate(
    input_variables=["strategies"],
    template="""
    Aggregate risk from:
    {strategies}

    Assess:
    1. Market risks
    2. Liquidity risks
    3. Black swan risks
    4. Final risk rating (1-5)
    """
)

# --- Portfolio Manager ---
portfolio_prompt = PromptTemplate(
    input_variables=["strategies", "risk_rating"],
    template="""
    Consolidate strategies:
    {strategies}

    Given risk rating {risk_rating}, determine:
    1. Final trade action (Buy, Sell, Short, Hold)
    2. Adjusted position sizing
    3. Expected returns
    """
)

# --- Pipeline ---
def run_ai_hedge_fund(ticker: str):
    market_data = get_market_data(ticker)
    sentiment = analyze_sentiment(get_news(ticker))
    macro = analyze_macro()

    strategies = {
        "Stan Druckenmiller": stan_druckenmiller_strategy(market_data, sentiment, macro),
        "Warren Buffett": warren_buffett_strategy(market_data, sentiment, macro)
    }

    risk_rating = llm.invoke(risk_prompt.format(strategies=strategies)).content
    final_decision = llm.invoke(portfolio_prompt.format(strategies=strategies, risk_rating=risk_rating)).content

    with open(f"{ticker}_analysis.txt", "w") as f:
        f.write(f"=== {ticker} AI Hedge Fund Analysis ===\n")
        f.write("\nMarket Data:\n" + str(market_data))
        f.write("\n\nSentiment Analysis:\n" + sentiment)
        f.write("\n\nMacro Analysis:\n" + macro)
        f.write("\n\nAgent Strategies:\n" + str(strategies))
        f.write("\n\nRisk Assessment:\n" + risk_rating)
        f.write("\n\nFinal Decision:\n" + final_decision)
    remove_think_tags(f"{ticker}_analysis.txt")    

# Run analysis
run_ai_hedge_fund("AAPL")
