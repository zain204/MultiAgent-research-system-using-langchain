from src.tools.tools import web_search,scrape_url


r = web_search.invoke("what is agentic ai")
print(r)


url = "https://en.wikipedia.org/wiki/Artificial_intelligence"

content = scrape_url.invoke({'url':url})

# print(content)