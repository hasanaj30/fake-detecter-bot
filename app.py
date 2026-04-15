import os
import re
import requests
import feedparser
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# =========================
# LOAD TOKEN
# =========================
load_dotenv("token.env")
BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    raise Exception("❌ BOT_TOKEN not found in token.env")

# =========================
# SAMPLE NEWS DATABASE (FALLBACK + MATCHING BASE)
# =========================
NEWS_DB = [
    {
        "title": "ISRO successfully launches PSLV satellite from Sriharikota",
        "link": "https://www.isro.gov.in/",
        "content": "ISRO launched PSLV carrying earth observation satellite for weather monitoring"
    },
    {
        "title": "RBI keeps repo rate unchanged in monetary policy meeting",
        "link": "https://rbi.org.in/",
        "content": "Reserve Bank of India monetary policy repo rate unchanged inflation growth"
    },
    {
        "title": "India expands digital education in rural schools",
        "link": "https://india.gov.in/",
        "content": "Government improves internet connectivity and smart classrooms rural education"
    },
    {
        "title": "India boosts renewable energy with solar and wind projects",
        "link": "https://mnre.gov.in/",
        "content": "India renewable energy solar wind power expansion climate policy"
    },
    {
        "title": "Indian economy shows steady growth in latest report",
        "link": "https://livemint.com/",
        "content": "India GDP growth strong financial report economy stable inflation control"
    }
]

# =========================
# RSS TOP NEWS (REAL TIME)
# =========================
def get_top_news():
    try:
        url = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)

        news = []
        for entry in feed.entries[:5]:
            news.append({
                "title": entry.title,
                "link": entry.link,
                "content": entry.title
            })
        return news if news else NEWS_DB[:5]

    except:
        return NEWS_DB[:5]

# =========================
# TEXT CLEANING
# =========================
def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text

# =========================
# AI INTELLIGENCE ENGINE v11
# =========================
def analyze_news(user_text):
    corpus = [clean(n["content"]) for n in NEWS_DB]
    corpus.append(clean(user_text))

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)

    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])[0]

    best_score = float(np.max(similarity))
    best_index = int(np.argmax(similarity))

    matched_news = NEWS_DB[best_index] if best_score > 0.25 else None

    # =========================
    # FAKE/REAL SCORING (FIXED LOGIC)
    # =========================
    base_real = best_score * 100
    base_fake = 100 - base_real

    # adjust logic for realistic behavior
    if best_score < 0.25:
        verdict = "🔴 LIKELY FAKE"
    elif best_score < 0.55:
        verdict = "🟡 UNCERTAIN"
    else:
        verdict = "🟢 LIKELY REAL"

    return {
        "verdict": verdict,
        "real": round(base_real, 2),
        "fake": round(base_fake, 2),
        "matched": matched_news
    }

# =========================
# FORMAT RESPONSE
# =========================
def format_report(text):
    result = analyze_news(text)
    top_news = get_top_news()

    reply = f"""
🧠 PRO AI INTELLIGENCE v11 ULTRA

📰 Input:
{text}

🔍 Verdict: {result['verdict']}

📊 Real Probability: {result['real']}%
📉 Fake Probability: {result['fake']}%

🧾 Analysis:
• Semantic AI matching enabled
• TF-IDF similarity scoring applied

📰 Related News:
"""

    if result["matched"]:
        reply += f"- {result['matched']['title']}\n{result['matched']['link']}\n"
    else:
        reply += "⚠️ No relevant news found\n"

    reply += "\n🔥 TOP 5 DAILY NEWS:\n"
    for i, n in enumerate(top_news, 1):
        reply += f"{i}. {n['title']}\n{n['link']}\n\n"

    reply += """
⚠️ DISCLAIMER:
This AI prediction is for reference only. Not 100% accurate.
"""
    return reply

# =========================
# TELEGRAM HANDLERS
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 PRO AI INTELLIGENCE v11 READY\nSend any news text to analyze.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    response = format_report(user_text)
    await update.message.reply_text(response)

# =========================
# MAIN
# =========================
def main():
    print("🚀 PRO AI INTELLIGENCE v11 RUNNING")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.run_polling()

if __name__ == "__main__":
    main()