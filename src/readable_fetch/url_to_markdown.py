"""
url_to_markdown.py: Returns readable text from web URLs.
"""

import requests
import argparse
from readability import Document # type: ignore
from markdownify import markdownify as md


def fetch_and_markdownify(url: str) -> str:
    """
    Fetches the content of a URL, extracts the main article, and converts it to Markdown.

    Args:
        url: The URL of the webpage to process.

    Returns:
        A string containing the article content in Markdown format,
        or an error message if the process fails.
    """
    try:
        # Step 1: Fetch the HTML content from the URL
        # We add a user-agent header to mimic a web browser, which can help avoid
        # being blocked by some websites.
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(url, headers=headers, timeout=10)

        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # Step 2: Extract the main article content using readability
        # The Document class takes the HTML content and parses it.
        doc: Document = Document(response.text) # type: ignore

        # The summary() method returns the main content as a clean HTML string.
        # The html_partial=True argument ensures we get just the article's body.
        clean_html = doc.summary(html_partial=True) # type: ignore

        # Step 3: Convert the clean HTML to Markdown
        # The markdownify function handles the conversion.
        # We can specify heading styles and other options if needed.
        markdown_content = md(clean_html, heading_style="ATX") # type: ignore

        return markdown_content

    except requests.exceptions.RequestException as e:
        return f"Error: Could not fetch the URL. Details: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def main():
    """
    Main function to handle command-line arguments and execute the script.
    """
    # Set up argument parser to accept a URL from the command line
    parser = argparse.ArgumentParser(
        description="Fetch a URL and convert its main content to Markdown."
    )
    parser.add_argument(
        "url", type=str, help="The full URL of the article you want to convert."
    )

    args = parser.parse_args()

    # Call the main function with the provided URL
    markdown_output = fetch_and_markdownify(args.url)

    # Print the result to the console
    print(markdown_output)


if __name__ == "__main__":
    main()
