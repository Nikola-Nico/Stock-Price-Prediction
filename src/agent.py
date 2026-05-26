import pandas as pd
import json
import os
from pydantic import BaseModel, Field
from openai import OpenAI

# Поставување на OpenAI клиентот преку OpenRouter
client = OpenAI(
    api_key="",
    base_url="https://openrouter.ai/api/v1"
)

# Вчитување на финансиските податоци со безбедни релативни патеки
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except:
    base_dir = "."

def load_csv_safely(filename):
    paths = [
        os.path.join(base_dir, '..', 'data', 'raw', filename),
        os.path.join(base_dir, 'data', 'raw', filename),
        f"data/raw/{filename}"
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return pd.DataFrame()

companies_df = load_csv_safely("sp500_companies.csv")
stocks_df = load_csv_safely("sp500_stocks.csv")

if not stocks_df.empty and 'Close' in stocks_df.columns:
    stocks_df["Close"] = pd.to_numeric(stocks_df["Close"], errors="coerce")
if not companies_df.empty and 'Weight' in companies_df.columns:
    companies_df["Weight"] = pd.to_numeric(companies_df["Weight"], errors="coerce")


# Pydantic модел за структурирана анализа на акции
class StockAnalysis(BaseModel):
    ticker: str = Field(description="The stock ticker symbol, e.g. AAPL")
    trend: str = Field(description="Bullish, Bearish, or Neutral based on recent data")
    summary: str = Field(description="One sentence financial summary or recommendation")


# ДЕФИНИЦИЈА НА АЛАТКИТЕ (TOOLS) - ОПТИМИЗИРАНА И ПАМЕТНА ВЕРЗИЈА
def get_top_companies_by_weight(n: int = 5) -> str:
    if companies_df.empty: 
        return "Податоците за компаниите не се достапни."
    
    # СЕПАКУВАЊЕ: Динамички проверуваме која колона за име постои во твојот CSV за да нема index error
    name_col = None
    for col in ["Security", "Name", "ShortName", "Company"]:
        if col in companies_df.columns:
            name_col = col
            break
            
    # Ги градиме достапните колони кои сигурно ги има во индексот
    available_cols = ["Symbol"]
    if name_col:
        available_cols.append(name_col)
    if "Sector" in companies_df.columns:
        available_cols.append("Sector")
    if "Weight" in companies_df.columns:
        available_cols.append("Weight")
        
    try:
        if "Weight" in companies_df.columns:
            res = companies_df.nlargest(n, "Weight")[available_cols]
        else:
            res = companies_df.head(n)[available_cols]
            
        return res.to_json(orient="records")
    except Exception as e:
        return json.dumps({"error": f"Грешка при филтрирање на компаниите: {str(e)}"})

def get_stock_stats(ticker: str) -> str:
    if stocks_df.empty: return "Податоците за акциите не се достапни."
    filtered = stocks_df[stocks_df["Symbol"].str.upper() == ticker.upper()]
    if filtered.empty:
        return json.dumps({"error": f"Тикерот {ticker} не е пронајден во базата."})
    last_30 = filtered.tail(30)
    latest_row = filtered.iloc[-1]
    return json.dumps({
        "ticker": ticker.upper(),
        "latest_close": round(latest_row["Close"], 2),
        "max_30d": round(last_30["Close"].max(), 2),
        "min_30d": round(last_30["Close"].min(), 2)
    })

def get_market_summary() -> str:
    if companies_df.empty: return "Нема достапни податоци."
    total_companies = len(companies_df["Symbol"].unique())
    sectors_count = companies_df["Sector"].value_counts().to_dict()
    return json.dumps({
        "total_tracked_companies": total_companies,
        "sector_distribution": sectors_count
    })


# ГЛАВНА ФУНКЦИЈА ЗА ПОВИКУВАЊЕ НА АГЕНТОТ
def ask_agent(user_query: str, xgb_predictor_func=None) -> str:
    """
    Ја извршува логиката на агентот. Прима опционална функција од app.py 
    за да ја изврши вистинската XGBoost прогноза кога моделот ќе ја побара.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_stats",
                "description": "Ги враќа poslednite историски цени, максимум и минимум во последните 30 дена за одреден тикер.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Пример: NVDA, AAPL"}
                    },
                    "required": ["ticker"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "predict_stock_price",
                "description": "Пресметува идна вредност и ја анализира состојбата преку XGBoost со технички индикатори (RSI, MA).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Пример: TSLA"}
                    },
                    "required": ["ticker"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_top_companies_by_weight",
                "description": "Враќа топ N највредни компании во S&P 500 индексот.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "description": "Број на компании"}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_market_summary",
                "description": "Дава секторски преглед на пазарот."
            }
        }
    ]

    system_prompt = (
        "Ти си професионален AI Финансиски Аналитичар и Senior Quant Strategist.\n"
        "Ако корисникот те праша за предвидување, раст, пад или иднина на некоја акција, ЗАДОЛЖИТЕЛНО повикај ја алатката 'predict_stock_price'.\n"
        "Кога ќе ги добиеш резултатите од алатките, детално анализирај го RSI (над 70 е прекупено, под 30 е препродадено) и подвижните просеци.\n"
        "Одговарај секогаш на македонски јазик, со јасни bullet points и биди прецизен со бројките."
    )

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            tools=tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        if tool_calls:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
                response_message
            ]
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "get_stock_stats":
                    function_response = get_stock_stats(ticker=function_args.get("ticker"))
                elif function_name == "predict_stock_price":
                    if xgb_predictor_func:
                        try:
                            res = xgb_predictor_func(function_args.get("ticker").upper(), days_to_predict=30)
                            price_diff = res['predicted_end_price'] - res['current_price']
                            pct_change = (price_diff / res['current_price']) * 100
                            function_response = json.dumps({
                                "ticker": function_args.get("ticker").upper(),
                                "current_price": res["current_price"],
                                "xgb_predicted_30d_price": res["predicted_end_price"],
                                "pct_change": round(pct_change, 2),
                                "rsi": res["metrics"]["RSI"],
                                "volatility": res["metrics"]["Volatility"],
                                "ma7": res["metrics"]["MA7"],
                                "ma21": res["metrics"]["ma21"] if "ma21" in res["metrics"] else res["metrics"].get("MA21", res["current_price"])
                            })
                        except Exception as e:
                            function_response = json.dumps({"error": str(e)})
                    else:
                        function_response = json.dumps({"ticker": function_args.get("ticker"), "msg": "XGBoost не е поврзан."})
                elif function_name == "get_top_companies_by_weight":
                    function_response = get_top_companies_by_weight(n=function_args.get("n", 5))
                elif function_name == "get_market_summary":
                    function_response = get_market_summary()
                else:
                    function_response = "Алатката не е пронајдена."
                    
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response
                })
            
            second_response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=messages
            )
            return second_response.choices[0].message.content
        else:
            return response_message.content
            
    except Exception as e:
        return f"Грешка при комуникација со AI асистентот: {str(e)}"

def get_structured_analysis(ticker: str) -> StockAnalysis:
    stats = json.loads(get_stock_stats(ticker))
    prompt = (
        f"Return ONLY a valid JSON object matching this schema: ticker, trend, summary. "
        f"Stock: {ticker}, Data: {json.dumps(stats)}"
    )
    try:
        res = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        raw = res.choices[0].message.content
        return StockAnalysis(**json.loads(raw))
    except Exception as e:
        return StockAnalysis(ticker=ticker, trend="Neutral", summary=f"Грешка при структурирана анализа: {e}")