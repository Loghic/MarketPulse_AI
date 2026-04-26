# News Sentiment Analysis

## Overview

News sentiment adjusts model predictions post-hoc. The idea: if k-NN says UP at 60% confidence, but today's news is overwhelmingly negative, maybe that 60% should be lower.

The pipeline:

```
yfinance news API → extract headlines → score with VADER or naive → sentiment_score ∈ [-1, 1]
```

## Scoring methods

### VADER (default)

NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based model specifically designed for social media and news text. It uses a curated lexicon of ~7,500 words with pre-assigned sentiment scores.

What makes VADER better than keyword matching:
- **Negation:** "not good" → negative (naive would count "good" as positive)
- **Intensifiers:** "VERY good" → more positive than "good"
- **Capitalization:** "AMAZING" → stronger than "amazing"
- **Conjunctions:** "good but not great" → mixed, slightly negative
- **Punctuation:** "great!!!" → stronger than "great"

We use the `compound` score, which combines positive, negative, and neutral into a single float in [-1, 1]. This is averaged across all headlines:

```python
score = mean([vader.polarity_scores(headline)["compound"] for headline in headlines])
```

**Dependency:** `nltk` package + `vader_lexicon` data (downloaded automatically on first use). If download fails, falls back to naive.

### Naive (fallback)

Simple keyword matching against two word lists (~18 positive, ~18 negative words). Each keyword match adds +1 or -1, normalized by headline count:

```python
score = max(-1, min(1, raw_count / len(headlines)))
```

Limitations:
- No context: "not good" → counts both "not" (nothing) and "good" (+1) = positive
- No intensity: "CRASH" and "crash" are the same
- Limited vocabulary: only matches exact words in the lists

Kept as a zero-dependency fallback and baseline for comparison.

## Sentiment integration with models

### The two-stage approach

Sentiment is NOT a training feature. It's applied after the model makes its prediction:

```
Step 1: model.predict(price_data) → direction, base_probability
Step 2: _apply_sentiment(direction, base_probability, sentiment_score) → adjusted_direction, adjusted_probability
```

Why not feed sentiment into training?
- We don't have historical daily sentiment scores. We only have today's news.
- Even if we scraped historical headlines, matching them to exact trading days is noisy.
- The two-stage approach is transparent: you can see exactly how much sentiment changed the prediction.

### The math

```python
SENTIMENT_WEIGHT = 0.20  # max shift: ±20 percentage points

# Convert to "probability of UP"
prob_up = prob if prediction == UP else (1 - prob)

# Apply shift
prob_up_adjusted = prob_up + sentiment_score × SENTIMENT_WEIGHT

# Derive final prediction
if prob_up_adjusted >= 0.5: → UP
else: → DOWN
```

Examples:
- k-NN says UP 60%, sentiment = 0 → UP 60% (no change)
- k-NN says UP 60%, sentiment = +0.5 → UP 70% (boosted)
- k-NN says UP 55%, sentiment = -1.0 → DOWN 65% (flipped!)

### The weight parameter

`SENTIMENT_WEIGHT = 0.20` is hardcoded in both model files. At 0.20:
- Neutral sentiment (0.0): no effect
- Mildly positive (+0.3): shifts by +6pp
- Strongly positive (+1.0): shifts by +20pp
- Strongly negative (-1.0): shifts by -20pp

This was chosen to be meaningful but not dominant. A perfect sentiment score can swing a borderline call (55% → 35%), but can't override a strong pattern signal (80% → 60%, still same direction).

## Caching

Headlines and scores are cached in SQLite for one calendar day. This prevents hitting the yfinance news API repeatedly during backtesting or multiple predictions for the same ticker.

```
DB table: news_sentiment (ticker, date, headline, sentiment_score)
Primary key: (ticker, date, headline) — deduplication by headline text
```

## Limitations

- **No historical sentiment.** Backtesting uses today's sentiment for all holdout days. This is unrealistic — in real trading, each day would have its own news. The effect is that backtest results for `+ News` variants are less reliable than non-news variants.
- **yfinance news is limited.** The API returns ~10 recent headlines, not a comprehensive feed. Coverage varies by ticker — popular stocks get more headlines.
- **VADER is general-purpose.** It wasn't trained on financial text specifically. "Short" (as in short-selling) is negative in VADER, but in finance it's a neutral action. FinBERT (on the roadmap) would be more accurate for financial sentiment.
- **Keyword naive is very rough.** The 36-word vocabulary misses most nuance. It's really just a baseline to compare VADER against.
