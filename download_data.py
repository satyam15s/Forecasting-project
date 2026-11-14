"""Download a fixed snapshot of daily S&P 500 closes."""

from csv import DictReader, DictWriter
from datetime import date
from io import TextIOWrapper
from math import isfinite, isclose
from pathlib import Path
from urllib.request import urlopen


DATA_PATH = Path(__file__).resolve().parent / "data" / "sp500_daily.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/fja05680/dow-sp500-100-years/"
    "52a1ddd481fc947e511f709896d4f2b3fb49fb0e/SP500.csv"
)
START = date(1979, 12, 31)  # Prior close for the first return in 1980
END = date(1987, 12, 31)
FRIDAY = date(1987, 10, 16)
BLACK_MONDAY = date(1987, 10, 19)


def main():
    closes = {}
    with urlopen(DATA_URL, timeout=30) as response:
        for row in DictReader(TextIOWrapper(response, encoding="utf-8")):
            day = date.fromisoformat(row["Date"])
            if START <= day <= END:
                close = float(row["Close"])
                if day in closes or not isfinite(close) or close <= 0:
                    raise ValueError(f"Duplicate session or invalid close on {day}")
                closes[day] = close

    # Cross-check the two event closes against Schwert (1989).
    expected = {FRIDAY: 282.70, BLACK_MONDAY: 224.84}
    if START not in closes or END not in closes or not all(
        day in closes and isclose(closes[day], close, abs_tol=0.005)
        for day, close in expected.items()
    ):
        raise ValueError("The source is missing expected S&P 500 closes.")

    DATA_PATH.parent.mkdir(exist_ok=True)
    with DATA_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = DictWriter(file, fieldnames=["date", "close"])
        writer.writeheader()
        writer.writerows(
            {"date": day, "close": close} for day, close in sorted(closes.items())
        )
    print(f"Saved {len(closes):,} daily closes to {DATA_PATH}")


if __name__ == "__main__":
    main()
