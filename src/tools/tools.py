from langchain.tools import tool
from dotenv import load_dotenv
import requests
import os
# from langchain_tavily import TavilySearch
from tavily import TavilyClient
from bs4 import BeautifulSoup
from readability import Document
import re
import trafilatura


load_dotenv()

### using langchain_tavily###
# tavily_search = TavilySearch(max_results=5)

# result = tavily_search.invoke({
#     'query':"what is agentic ai in 2 lines"
# })

# print(result)



###using tavily-python###

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
@tool
def web_search(query:str)->str:

    """Search the web for relevant information about a given topic or question.

    Use this tool when you need to find online information,
    current facts, or relevant webpages.

    Args:
        query: The topic or question to search for."""

    result = tavily.search(query=query,max_results=2)

    out=[]

    for r in result['results']:
        out.append(
            f"Title : {r['title']} \n URL: {r['url']} \n Snippet: {r['content'][:300]}"
        )

    return "\n----\n".join(out)    
   


### scrape url ###

@tool
def scrape_url(url: str) -> str:
    """Scrape a URL and extract its main text content."""

    try:
        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            return f"Could not download content from: {url}"

        content = trafilatura.extract(
            downloaded,
            include_links=False,
            include_images=False
        )

        if not content:
            return f"Could not extract readable content from: {url}"

        return content

    except Exception as e:
        return f"Error scraping URL: {e}"
