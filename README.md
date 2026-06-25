# Netflix Shows — Data Engineering & EDA

## Project Overview

This project works with the [Netflix Movies and TV Shows dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows/data) from Kaggle. The dataset was loaded programmatically using `kagglehub` inside Google Colab — no manual downloads or GUI uploads were used, in line with the assignment's reproducibility requirement.

This README covers **Part 1: The Data Engineering Pipeline** — the cleaning and feature-engineering work that had to happen before any visualization could be trusted. The guiding principle throughout was simple: *if the data isn't cleaned properly, the visualizations built on top of it will be wrong.* So every step below wasn't just "run code and move on" — it involved checking, breaking, and re-checking the data until the result could actually be trusted.

---

## 1. The Duration Fix

### The Problem
The `duration` column in the raw dataset is a single string field that means two completely different things depending on content type:
- For Movies: something like `"90 min"`
- For TV Shows: something like `"2 Seasons"`

Mixing these into one column makes any numeric analysis (averages, distributions, comparisons) meaningless — a "duration" of `2` could mean 2 minutes or 2 seasons, and there's no way to tell without checking `type` separately every single time.

### The Approach
Rather than guessing which row was a Movie or TV Show from the `duration` string itself, the existing `type` column (which already cleanly states `"Movie"` or `"TV Show"`) was used to split the numeric value into two dedicated, type-specific columns:

```python
df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float)

df['duration_minutes'] = np.where(df['type'] == 'Movie', df['duration_num'], np.nan)
df['duration_seasons'] = np.where(df['type'] == 'TV Show', df['duration_num'], np.nan)

df.drop(columns=['duration_num'], inplace=True)
```

### Why `NaN` and not `0`
This was a deliberate decision, not a default. A Movie doesn't have "zero seasons" — the concept of seasons simply doesn't apply to it. Using `0` would have been factually wrong and dangerous for later statistics: any `.mean()` or `.sum()` on `duration_seasons` would have silently included a flood of fake zeros from every movie in the dataset, dragging down the average and producing a misleading result. `NaN` correctly represents "not applicable," and pandas automatically excludes `NaN` from aggregate functions like `.mean()` and `.describe()` — so the stats stay honest.

`duration_num` was kept only as a temporary scratch column to extract the digits before splitting, and was dropped immediately afterward since keeping it around would have meant duplicate, unused data sitting in the dataframe — clutter that could easily get double-counted by mistake in a careless future groupby.

### A Real Data Quality Bug: The Rating/Duration Swap
After running the extraction, 3 rows came back with missing `duration` values. Rather than dropping them or assuming they were "just missing," they were investigated directly:

```python
df[df['duration'].isnull()][['title', 'type', 'duration', 'rating']]
```

This revealed a genuine data artifact in the source CSV: for these 3 titles (all Louis C.K. specials), the `rating` and `duration` values had been swapped — `rating` contained values like `"74 min"`, while `duration` was empty. This is a known quirk in this dataset, caused by a parsing misalignment somewhere upstream of the CSV.

**The fix** — a vectorized swap, done without a loop:

```python
mask = df['duration'].isnull()
df.loc[mask, ['duration', 'rating']] = df.loc[mask, ['rating', 'duration']].values
```

The `.values` part is essential here — without it, pandas tries to align the assignment by column name during the swap, which causes both columns to end up with the same value instead of actually swapping. Using `.values` strips the column labels and does a raw positional copy, which is what makes the swap work correctly in one line.

Since the true original `rating` for these 3 rows is unrecoverable, it was deliberately set to `NaN` rather than guessed — an honest "missing" is better than a fabricated value.

**Lesson learned:** always investigate *why* a value is missing before deciding how to handle it. A `NaN` can mean "truly absent" or it can mean "the value moved somewhere else due to a data error" — and those two cases require completely different fixes.

### A Notebook-State Mistake (and what it taught)
While debugging the swap, a confusing situation came up: a verification query returned a row where `duration` showed `"Movie"` and `rating` showed `"74 min"` — clearly wrong, since `type` was never part of the swap logic at all.

The cause: in Jupyter/Colab, all cells share one live, persistent `df` in memory, and cells run in the order they're *executed*, not the order they appear on the page. Re-running an old cell, or running cells out of sequence, can leave the dataframe in an inconsistent state that doesn't match what the visible code says it should be.

**The fix:** `Runtime → Restart Runtime`, followed by running every cell once, top to bottom, in the correct final order. This guarantees the dataframe's actual state matches the notebook's code exactly — not a leftover mix of old experiments.

**Lesson learned:** any cell that does `drop()`, in-place mutation, or column overwrites is effectively "single-use." Re-running it on already-modified data will either throw an error (e.g. a `KeyError` for a column already dropped) or, worse, silently corrupt the data without throwing any error at all. From this point on, every "destructive" cell was treated as a one-time operation, verified once, and never blindly re-run without a full restart.

---

## 2. Unnesting Arrays (`cast`, `director`, `country`, `listed_in`)

### The Problem
Four columns store multiple values as a single comma-separated string, e.g.:
```
country: "United States, India, France"
listed_in: "Dramas, International Movies"
```
Left in this form, accurate counting is impossible — for example, "how many titles is each actor in" can't be computed while an actor's name is buried inside a string shared with several other names.

### The Approach
For each of the four columns, a separate exploded dataframe was created — deliberately **not** overwriting the main `df`, since the original one-row-per-title structure is still needed for most other analysis. Each column got the same repeatable pattern:

```python
df_country = df[['title', 'country']].dropna(subset=['country']).copy()
df_country['country'] = df_country['country'].str.split(', ')
df_country = df_country.explode('country')
df_country['country'] = df_country['country'].str.strip()
```

The same pattern was applied to `cast`, `director`, and `listed_in`.

### Why `.copy()` Matters
Slicing a dataframe like `df[['title', 'country']]` can return a *view* rather than a guaranteed independent copy. Modifying that slice afterward can trigger a `SettingWithCopyWarning`, because pandas can't always be sure whether the original `df` or the new slice was meant to be changed. Adding `.copy()` removes that ambiguity entirely and makes the independence explicit — even on pandas versions where the warning doesn't always fire, it's worth doing as a habit rather than relying on pandas' internal guesswork.

### Why `.split(', ')` *and* `.strip()` — Not Just One
`str.split(', ')` (comma plus space) correctly handles the expected, well-formatted case. But real-world data isn't guaranteed to be perfectly consistent — a row with `"USA,India"` (no space) or `"USA,  India"` (double space) would either fail to split cleanly or leave stray whitespace. `.strip()` was added afterward as a safety net: it removes leading/trailing whitespace regardless of how the string was actually formatted, so the result stays clean even if the source data isn't perfectly uniform.

This was actually verified rather than assumed:
```python
df['country'].dropna().str.contains(',\S').sum()
```
This returned `0`, confirming the source data had no comma-without-space inconsistencies. Good to know — but `.strip()` was kept anyway as cheap insurance against any other column having different formatting quirks.

### Handling Missing Values in These Columns
Some titles have no `country`, `cast`, or `director` listed. Rather than guessing or fabricating a value, the missing rows were simply excluded *only from the column-specific exploded dataframe* used for that one type of analysis (e.g. excluded from `df_country` for country-based charts), while remaining fully intact in the main `df` for every other type of analysis (genre charts, actor charts, etc.). Dropping a row from the main dataset entirely just because one field was missing would have thrown away perfectly good data needed elsewhere.

The percentage of missing `country` values was checked explicitly before deciding this:
```python
df['country'].isnull().mean() * 100
```
This kind of percentage is documented here rather than silently ignored, since it directly affects how much of the dataset country-based visualizations actually represent.

---

## 3. Datetime Parsing (`date_added`)

### The Approach
```python
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month
df['day_added'] = df['date_added'].dt.day_name()
```

`errors='coerce'` was used deliberately: it tells pandas that any value it can't parse as a valid date should become `NaT` (datetime's version of `NaN`) instead of crashing the entire conversion. Without it, a single malformed date string anywhere in 8,000+ rows would have broken the whole pipeline.

### Investigating the Missing Dates
After parsing, 98 rows came back with a missing `date_added`. Rather than assuming this was either "obviously fine" or "obviously a bug," it was investigated directly by checking what the *original, unparsed* string actually looked like for those specific rows:

```python
mask = df['date_added'].isnull()
df[mask]['date_added'].unique()[:20]
```

This returned a single result: `array([nan], dtype=object)` — meaning every one of these 98 rows was genuinely blank in the original source CSV, with no inconsistent formatting or recoverable value hiding underneath. This was a clean, confirmed case of true missing data, not a parsing failure caused by a date format pandas didn't understand.

**Why this check mattered:** a missing-date count alone doesn't tell you *why* the dates are missing. It could mean "blank in the source" (nothing to do) or "a date existed but in an unparseable format" (something to fix). Confirming which one it was — by going back to the raw string before any parsing was applied — turned an assumption into a verified fact.

These 98 rows keep `NaT` across `date_added`, `year_added`, `month_added`, and `day_added`, and will be excluded from any time-based visualizations in Part 2, since there's no honest way to plot a date that was never recorded.

---

## Summary of Key Decisions & Lessons

| Issue | Decision Made | Why |
|---|---|---|
| Movie vs. TV Show duration mixed in one column | Split into `duration_minutes` and `duration_seasons` using `type` | One column meaning two different things makes numeric analysis meaningless |
| Non-applicable duration values | Filled with `NaN`, not `0` | `0` would silently corrupt averages and sums; `NaN` is correctly excluded from aggregates |
| 3 rows with swapped `rating`/`duration` values | Vectorized `.loc` swap using `.values` | A real upstream data bug, not a missing-value case — fixed at the source rather than worked around |
| Unrecoverable original `rating` for swapped rows | Set to `NaN` | Honest about not knowing, rather than guessing a value |
| Notebook state corruption from re-running destructive cells | Restart Runtime + run all cells once, in order | Jupyter/Colab share one live `df` across cells regardless of page order; re-running drop/mutation cells silently corrupts state |
| Comma-separated multi-value columns | Exploded into separate dataframes, main `df` left untouched | Original one-row-per-title structure still needed for most other analysis |
| Possible whitespace inconsistency after splitting | Used `.split(', ')` + `.strip()` together, verified assumption with a direct check | Don't trust source data formatting blindly, even when split appears to work |
| Missing values in `country`/`cast`/`director` | Dropped only from the relevant exploded dataframe, kept in main `df` | Avoids fabricating data while not discarding rows useful elsewhere |
| 98 missing `date_added` values | Verified against raw pre-parse strings before deciding how to treat them | Confirmed genuine missing data rather than a parsing failure — different problems need different fixes |

The overarching theme across all of Part 1: every fix was made only after directly inspecting the actual rows involved, rather than assuming what a `NaN` or an error meant. Several "obvious" first guesses turned out to be incomplete or wrong once actually checked against the data — which is exactly the kind of discipline this assignment was designed to build before any visualization work begins.

---

*Part 2 (Visualizations) will be documented separately once the visualization pipeline is complete.*
