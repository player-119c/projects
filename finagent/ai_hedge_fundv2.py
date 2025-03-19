from langchain_core.runnables import RunnablePassthrough, RunnableParallel,RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from apikey import GROQ_KEY, ALPHAVANTAGE_API_KEY,NEWS_API_KEY_4
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
from newsapi.newsapi_client import NewsApiClient
import yfinance as yf

llm= ChatGroq(model='deepseek-r1-distill-qwen-32b',api_key=GROQ_KEY,temperature=0.7)
newsapi=NewsApiClient(api_key=NEWS_API_KEY_4)
# alphavantage=AlphaVantageAPIWrapper(alphavantage_api_key=ALPHAVANTAGE_API_KEY)


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
        "beta": stock.info.get('beta', 'N/A')  # Get beta directly from yfinance
    }

def get_news(ticker: str) -> list:
    return newsapi.get_everything(
        q=f"{ticker} stock",
        language='en',
        sort_by='relevancy',
        page_size=5
    )['articles']

# --- Analysis Components ---
def analyze_sentiment(articles: list) -> str:
    news_text = "\n".join([f"{a['title']}: {a['description']}" for a in articles])
    return llm.invoke(f"Analyze market sentiment from:\n{news_text}").content

def analyze_macro(ticker: str) -> str:  # Add ticker parameter
    news = newsapi.get_top_headlines(
        q="economy GDP inflation interest rates unemployment",  # More relevant query
        category='business',
        page_size=5
    )['articles']
    news_text = "\n".join([f"{a['title']}: {a['description']}" for a in news])
    return llm.invoke(f"Analyze macroeconomic trends from:\n{news_text}").content




# --- Strategy Components ---
quant_prompt = PromptTemplate(
    input_variables=["market_data", "sentiment", "macro"],
    template="""
    Develop quantitative strategy using:
    Market Data: {market_data}
    Sentiment: {sentiment}
    Macro: {macro}
    
    Include:
    1. Entry/exit levels
    2. Position sizing
    3. Technical indicators
    4. Risk management
    5. Return projections
    """
)

risk_prompt = PromptTemplate(
    input_variables=["strategy"],
    template="""
    Assess risks for:
    {strategy}
    
    Evaluate:
    1. Market risks
    2. Liquidity risks
    3. Black swan risks
    4. Final risk rating (1-5)
    """
)



# --- Pipeline ---
analysis_chain = (
    RunnableParallel(
        market_data=RunnableLambda(get_market_data),
        sentiment=(
            RunnableLambda(get_news)
            | RunnableLambda(analyze_sentiment)
        ),
        macro=RunnableLambda(analyze_macro)
    )
    | RunnablePassthrough.assign(strategy=quant_prompt | llm)
    | RunnablePassthrough.assign(risk_assessment=risk_prompt | llm)
)


# --- Execution ---
def run_ai_hedge_fund(ticker: str):
    result = analysis_chain.invoke(ticker)
    
    with open(f"{ticker}_analysis.txt", "w") as f:
        f.write(f"=== {ticker} Analysis Report ===\n")
        f.write("\nMarket Data:\n" + str(result['market_data']))
        f.write("\n\nSentiment Analysis:\n" + result['sentiment'])
        f.write("\n\nMacro Analysis:\n" + result['macro'])
        f.write("\n\nTrading Strategy:\n" + result['strategy'].content)
        f.write("\n\nRisk Assessment:\n" + result['risk_assessment'].content)


# Run analysis
run_ai_hedge_fund("MSFT")