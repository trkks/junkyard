#!/usr/bin/env python3
"""
Klikinsäästäjä-ng
To install dependencies:
    $ pip install edgegpt-fork newspaper4k playwrigth markdownify sqlalchemy sqlalchemy-utils
    $ playwright install firefox

usage: klikinsäästäjä.py [-h] <url|test>
"""

import asyncio
from enum import Enum
import json
import os
import random
import re
from typing import Dict, List, NamedTuple, Union
import newspaper
from playwright.sync_api import Browser, sync_playwright, TimeoutError

import requests

from jinja2 import Template
from EdgeGPT.EdgeGPT import Chatbot, ConversationStyle

from sqlalchemy import Column, Integer, String, Text, Float
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

from platformdirs import user_data_dir
from pathlib import Path

# Jinja template for the prompt
instructions = r"""
Generate a descriptive and unbiased news title from the news article context.
- Follow practices used in scientific writing.
- Include most important and interesting information in it.
- Title should not be clickbait.
- If article is based on content of intrest groups or people who has vested interest ot topic title must indicate that.
- If original title is good enough, close of it or you are not sure how to improve it based on context, use it as is.
- Do NOT generate a new title for comments, opinion pieces, reviews, clearly marked sponsored content, or other articles that are not meant to be objective.
- If article is not news report, mention type of the content in the title.
- Provide reasoning for the new title, and issues with the original title.
- Keep the title concise and under 255 characters.
- Use a same language for a title that the original news article is written in.
- Make estimation how clickbaity the old title is on a scale from 0.0 to 1.0.
- If article contains comments from users, ignore them.
- Article URL: {{original_url|escape}}
- Original title: {{_title|striptags|escape}}

Format response in json following this structure:
```json
{
    'title': {title},
    'reasoning': [{reasoning}, {reasoning}, ...],
    'clickbaitiness score': {clickbaitiness}
}
```
"""

load_dotenv()

# Get the path to the user's appdata directory
appdata_dir = Path(user_data_dir('klikinsäästäjä-ng'))
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


edgetgpt_bot = contextvars.ContextVar(f"{__name__}_edgetgpt_bot")
db_session = contextvars.ContextVar(f"{__name__}_db_session")

BaseModel = declarative_base()

# Monekypatch get_location_hint_from_locale to support Finnish
from EdgeGPT.utilities import get_location_hint_from_locale as _get_location_hint_from_locale  # noqa: E402
import EdgeGPT.request  # noqa: E402


# Add support for Finnish. Enums cannot be modified, so we need to create a new one.
class PatchedLocationHint(Enum):
    FI = {
        "locale": "fi-FI",
        "LocationHint": [
            {
                "country": "Finland",
                "state": "",
                "city": "Helsinki",
                "timezoneoffset": 2,
                "countryConfidence": 8,
                "Center": {
                    "Latitude": 60.1699,
                    "Longitude": 24.9384,
                },
                "RegionType": 2,
                "SourceType": 1,
            },
        ],
    }


def _patched_get_location_hint_from_locale(locale: str):
    """
    Gets the location hint from the locale.

    This is a patched version of the original function to ad support for Finnish.
    """
    # Fi-fi -> fi-FI
    _region, _locale = locale.split("-", 1)
    locale = f"{_region.lower()}-{_locale.upper()}"

    # Find the location hint from the locale
    hint = next((hint for hint in PatchedLocationHint if hint.value["locale"] == locale), None)
    if hint is None:
        # Fallback to original function
        return _get_location_hint_from_locale(locale)

    return hint.value["LocationHint"]


# Replace the original function with the patched one
EdgeGPT.request.get_location_hint_from_locale = _patched_get_location_hint_from_locale


class Href(NamedTuple):
    url: str
    title: str


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
    sensationalism: float = Column(Float)  # Sensationalism score of the original title

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


def get_db_session():
    session = db_session.get(None)
    if session is None:
        fallback_path = Path(appdata_dir, "hrefs.db")
        db_url = os.environ.get("DATABASE_URL", f"sqlite+pysqlite:///{fallback_path}")
        engine = create_engine(db_url, connect_args={"check_same_thread": False})

        if not database_exists(engine.url):
            logger.debug("Creating new database with url %r", db_url)
            create_database(engine.url)
        else:
            logger.debug("Using existing database: %r", db_url)

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

    article_links = []

    for article in data["response"]:
        # Skip content that has sponsored content metadata
        if article.get('metadata', {}).get('sponsored_content', False):
            logger.debug("Skipping sponsored content: %r", article['title'])
            continue

        url = base_url.format(**article)
        article_links.append(Href(url, article['title']))

    return article_links


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
        locale="fi-FI",
        webpage_context=webpage_context,
        no_search=False,
        mode="gpt4-turbo",
    )
    logger.debug(response)
    #print(response)
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


def fetch_page_html(url, browser: Browser):
    """
    Fetches the html of the news article page.
    """

    # Remove javascript warnings and errors produced by browser
    def _filter(record):
        if record.levelno >= logging.INFO:
            if record.msg.startswith("[JavaScript Warning:") or record.msg.startswith("[JavaScript Error:"):
                return False

        return True

    # FIXME: Filter might be added multiple times
    _logger = logging.getLogger("playwright.browser")
    _logger.addFilter(_filter)

    def console_log(msg):
        """
        Log the playwright ConsoleMessage objects.
        """

        level_map = {
            "log": logging.DEBUG,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "info": logging.INFO,
            "assert": logging.ERROR,
            "debug": logging.DEBUG,
        }
        if msg.type not in level_map:
            return

        # Construct log record
        record = logging.LogRecord(
            name="playwright",
            level=level_map[msg.type],
            pathname=msg.location["url"],
            lineno=msg.location["lineNumber"],
            msg=msg.text,
            args=(),
            exc_info=None,
        )

        # Log the record
        _logger.handle(record)

    page = browser.new_page()
    page.on("console", console_log)

    page.goto(url)
    # Ignore timout errors
    try:
        # Prevent media from autoplaying
        page.add_script_tag(content=r"document.querySelectorAll('video').forEach((v) => { v.pause(); });")
        # Wait for the page to load
        page.wait_for_load_state("networkidle")
    except TimeoutError as e:
        logger.debug("Timout error while loading page %r: %r", url, e)

    # Convert relative links to absolute links
    page.add_script_tag(
        content=r"""
        document.querySelectorAll('a').forEach((a) => {
            if (a.getAttribute("href").startsWith('/')) {
                a.href = new URL(a.href, window.location.origin).href;
            }
        });
        """
    )

    html_content = page.content()

    # Validate content length
    content_length = len(html_content)
    logger.debug("Content length: %r", content_length)
    if content_length < 100:
        logger.debug("Page content: %r", html_content)
        raise Exception("Content length is too short")

    page.close()

    article = newspaper.article(url)
    article.download(html_content).parse()

    return article


def build(url):

    with sync_playwright() as playwright:
        # TODO: Change into edge
        browser = playwright.firefox.launch(headless=True, firefox_user_prefs={
            "intl.accept_languages": "fi",
            "media.autoplay.default": 0,
        })
        browser_context = browser.new_context()

        article = fetch_page_html(url, browser=browser)

        content_length = len(article.article_html)
        logger.debug("Content length: %r", content_length)
        if content_length < 100:
            raise Exception("Content length is too short")

        cookies = browser_context.cookies()
        logger.debug("Browser cookies: %r", cookies, extra={'cookies': cookies})

        browser_context.close()
        browser.close()

    prompt = generate_bot_prompt(article)
    context = md(article.article_html, heading_style="ATX").strip()

    print(context)

    truncated_context = repr(context[:150]) + " ... " + repr(context[-100:]) if len(context) > 255 else context
    logger.debug("Extracted context: %r", truncated_context, extra={'markup': True, 'context': context})

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
        sensationalism=bot_suggestion['clickbaitiness score'],

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
        'sensationalism': data.sensationalism,
        'reasoning': data.reasoning,
    })


if __name__ == "__main__":
    # Get url from command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Generate a title for a news article")
    parser.add_argument("url", help="URL of the news article. Use 'test' to fetch a random article from Iltalehti.", nargs="?", default="test")
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
            'sensationalism': data.sensationalism,
            'reasoning': data.reasoning,
        })
