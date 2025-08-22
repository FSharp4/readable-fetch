"""
Demonstration script for url_to_markdown.
"""

from src.readable_fetch.url_to_markdown import fetch_and_markdownify


def main():
    """
    Main function to run the demo.
    """
    # The URL to be converted to Markdown
    url = "https://en.wikipedia.org/wiki/Very-large-scale_integration"

    # Fetch and convert the URL content
    result = fetch_and_markdownify(url)

    print(result)


if __name__ == "__main__":
    main()
