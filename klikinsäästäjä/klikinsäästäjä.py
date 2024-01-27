#!/usr/bin/env python3
"""
Klikinsäästäjä-ng
To install dependencies:
    $ pip install edgegpt-fork newspaper4k playwrigth markdownify sqlalchemy sqlalchemy-utils
    $ playwright install firefox

usage: klikinsäästäjä.py [-h] <url|test>
"""

import asyncio
from dataclasses import dataclass
import json
import os
import random
import re
from typing import Dict, List, NamedTuple, Union
import newspaper
from playwright.sync_api import Browser, sync_playwright
import requests

from jinja2 import Template
from EdgeGPT.EdgeGPT import Chatbot, ConversationStyle

from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import DateTime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy_utils import database_exists, create_database
from markdownify import markdownify as md

from datetime import datetime
import logging
import contextvars
from dotenv import load_dotenv

from platformdirs import site_data_dir
from pathlib import Path

# Jinja template for the prompt
instructions = r"""
Generate a descriptive and unbiased news title from following news article context.
- Follow practices used in scientific writing.
- Include most important and interesting information in it.
- Title should not be clickbait.
- Use a same language for title that the news article is written in.
- If original title is good enough, use it as is.
- Provide reasoning for the title in English, and issues with the original title.
- Keep the title concise and under 255 characters.
- Article URL: {{original_url}}
- Original title: {{_title|striptags|escape}}

Format response in json following this structure:
```json
{
    'title': {title}
    'reasoning': [{reasoning}, {reasoning}, ...]
}
```
"""

load_dotenv()

# Get the path to the user's appdata directory
appdata_dir = Path(site_data_dir('klikinsäästäjä-ng'))
appdata_dir.mkdir(parents=True, exist_ok=True)

try:
    from rich.logging import RichHandler
    from rich.console import Console
    console = Console()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(message)s',
                        handlers=[RichHandler(console=console)])
    print = console.print

except ImportError:
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)


class Href(NamedTuple):
    url: str
    title: str


BaseModel = declarative_base()


class HrefModel(BaseModel):
    """
    Represents an article in the database.
    """

    __tablename__ = 'hrefs'

    id: int = Column(Integer, primary_key=True)  # Unique identifier for the article
    url: str = Column(String(255), index=True)  # URL of the article
    title: str = Column(String(255))  # Title of the article
    original_url: str = Column(String(255), index=True)  # URL to which the article is redirected
    original_title: str = Column(String(255))  # Original title of the article
    context: str = Column(Text)  # Content of the article

    _reasoning: str = Column(Text, name="reasoning")  # Reasoning for the title as a list

    created = Column(DateTime, default=datetime.utcnow)  # Date and time when the article was created
    modified = Column(DateTime, onupdate=datetime.utcnow)  # Date and time when the article was last modified
    published = Column(DateTime)  # Date and time when the article was published

    @property
    def reasoning(self):
        if self._reasoning is None:
            return None
        return json.loads(self._reasoning)

    @reasoning.setter
    def reasoning(self, value):
        self._reasoning = json.dumps(value)

    def __repr__(self):
        """
        Returns a string representation of the Article object.
        """
        return f"<Article(url={self.url!r}, title={self.title!r}, reasoning={self.reasoning!r})>"


db_session = contextvars.ContextVar(f"{__name__}_db_session")


def get_db_session():
    session = db_session.get(None)
    if session is None:
        fallback_path = Path(appdata_dir, "hrefs.db")
        db_url = os.environ.get("DATABASE_URL", f"sqlite+pysqlite:///{fallback_path}")
        engine = create_engine(db_url)

        if not database_exists(engine.url):
            logger.debug("Creating new database session with url %r", db_url)
            create_database(engine.url)
        else:
            logger.debug("Using existing database session with url %r", db_url)

        BaseModel.metadata.create_all(engine)

        Session = sessionmaker(bind=engine)
        session = Session()

        db_session.set(session)
    return session


def fetch_latest_iltalehti() -> List[Href]:
    latest_url = r"https://api.il.fi/v1/articles/iltalehti/lists/latest?limit=30&image_sizes[]=size138"
    base_url = r"https://www.iltalehti.fi/{category[category_name]}/a/{article_id}"

    response = requests.get(latest_url)
    data = response.json()

    urls = []

    for article in data["response"]:
        url = base_url.format(**article)
        urls.append(Href(url, article['title']))

    return urls


def login_to_helsingin_sanomat(browser):
    hs_username = os.environ["HS_USERNAME"]
    hs_password = os.environ["HS_PASSWORD"]

    browser.goto("https://www.hs.fi")
    browser.click("text=Kirjaudu")
    browser.fill("input[id='username']", hs_username)
    browser.fill("input[id='password']", hs_password)
    browser.click("text=Kirjaudu")
    browser.wait_for_load_state("networkidle")


def fetch_lastest_helsingin_sanomat() -> List[Href]:
    ...


def generate_bot_prompt(article: newspaper.Article):
    """
    Generates a prompt for the article.
    """
    # Add spaces to front of every line in context

    prompt = Template(instructions).render(**article.__dict__)

    return prompt


async def async_invoke_bot(prompt: str, webpage_context: str = None):
    bot = await Chatbot.create()
    response = await bot.ask(
        prompt=prompt,
        conversation_style=ConversationStyle.precise,
        simplify_response=True,
        locale="fi",
        webpage_context=webpage_context,
        no_search=True,
    )
    logger.debug(response)
    await bot.close()
    return response['text']


def query_bot_suggestion(prompt: str, webpage_context: str = None):
    """ Run async_invoke_bot() as a synchronous function """
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(async_invoke_bot(prompt, webpage_context))
    return response


def parse_bot_response(response) -> Dict[str, Union[str, list[str]]]:
    """
    Parses the response from the bot.
    """
    # Get data from md response block
    text = re.split(r"```json\n(.*)\n```", response, flags=re.MULTILINE | re.DOTALL)[1]
    data = json.loads(text)
    return data


def fetch_news_article(url, browser: Browser):
    page = browser.new_page()
    page.goto(url)
    html_content = page.content()

    page.close()

    article = newspaper.article(url)
    article.download(html_content).parse()

    return article


def build(url):

    with sync_playwright() as playwright:
        # TODO: Change into edge
        browser = playwright.firefox.launch(headless=True)
        browser_context = browser.new_context()

        article = fetch_news_article(url, browser=browser)

        # console.log(browser_context.cookies())

        browser_context.close()
        browser.close()

    prompt = generate_bot_prompt(article)
    context = md(article.article_html)
    bot_response = query_bot_suggestion(prompt, context)
    bot_suggestion = parse_bot_response(bot_response)

    # Create a new HrefModel object
    article = HrefModel(
        url=article.url,
        title=bot_suggestion['title'],
        original_url=article.original_url,
        original_title=article.title,
        context=context,
        reasoning=bot_suggestion['reasoning'],

        published=article.publish_date
    )
    # x = HrefModel(url="url", title="title", original_url="original_url", original_title="original_title", context="context", reasoning=["reasoning"], published=datetime.utcnow())

    return article


def get_href_by_url(url):
    session = get_db_session()
    article = session.query(HrefModel).filter_by(original_url=url).first()
    if article is None:
        article = build(url)
        session.add(article)
        session.commit()

    logger.debug(article)

    return article


def test():
    logger.info("Fetching latest news from Iltalehti")
    uutiset = fetch_latest_iltalehti()
    logger.debug(uutiset)
    # Select a random article
    uutinen = random.choice(uutiset)

    logger.info("Generating title for {%r}", uutinen)
    data = get_href_by_url(uutinen.url)
    print({
        'url': data.url,
        'href': data.original_url,
        'Original title': data.original_title,
        'New title': data.title,
        'reasoning': data.reasoning
    })


if __name__ == "__main__":
    # Get url from command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Generate a title for a news article")
    parser.add_argument("url", help="URL of the news article. Use 'test' to fetch a random article from Iltalehti.")
    args = parser.parse_args()

    if args.url == "test":
        test()
    else:
        data = get_href_by_url(args.url)
        print({
            'url': data.url,
            'href': data.original_url,
            'Original title': data.original_title,
            'New title': data.title,
            'reasoning': data.reasoning
        })
