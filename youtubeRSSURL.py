import requests
from bs4 import BeautifulSoup
import re

def get_youtube_rss(channel_url):
    try:
        # 1. Fetch the channel page content
        response = requests.get(channel_url)
        response.raise_for_status()
        
        # 2. Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 3. Look for the RSS link tag in the header
        # YouTube provides this link for discovery by RSS readers
        rss_link = soup.find('link', type="application/rss+xml")
        
        if rss_link and rss_link.get('href'):
            return rss_link.get('href')
        
        # 4. Fallback: Manual ID extraction if the link tag isn't found
        # Search for the "UC..." channel ID pattern in the source
        match = re.search(r'/(UC[a-zA-Z0-9_-]{22})', response.text)
        if match:
            channel_id = match.group(1)
            return f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
            
        return "Could not find RSS feed. Ensure the URL is a valid YouTube channel."

    except Exception as e:
        return f"Error: {e}"

# Example usage:
url = "https://youtube.com" # Replace with any channel URL
print(f"RSS Feed URL: {get_youtube_rss(url)}")


