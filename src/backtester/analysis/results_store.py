"""Results storage for strategy scans using DuckDB.

Enables:
1. Storing scan results across multiple tickers
2. Comparing performance across assets
3. Building portfolio from best strategies per ticker
"""

import json
from pathlib import Path

import duckdb
import polars as pl

from backtester.analysis.scanner import ScanSummary


class ResultsStore:
    """DuckDB-backed storage for strategy scan results.

    Stores results in a single table with columns for params, metrics, and periods.
    Enables cross-ticker analysis and portfolio construction.

    Example:
        store = ResultsStore("data/scan_results.duckdb")

        # Store results from multiple tickers
        for ticker in ["SPY", "QQQ", "IWM"]:
            summary = scan_ma_crossover(ticker, ...)
            store.save_scan(summary)

        # Compare best params across tickers
        best = store.best_per_ticker(metric="test_sharpe", n=3)

        # Get portfolio candidates
        candidates = store.portfolio_candidates(min_test_sharpe=0.5)
    """

    def __init__(self, db_path: str | Path = "data/scan_results.duckdb"):
        """Initialize the results store.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_tables()

    def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_results (
                ticker VARCHAR NOT NULL,
                train_start DATE NOT NULL,
                train_end DATE NOT NULL,
                test_start DATE NOT NULL,
                test_end DATE NOT NULL,
                params JSON NOT NULL,
                train_sharpe DOUBLE,
                train_return DOUBLE,
                train_max_dd DOUBLE,
                train_num_trades INTEGER,
                train_win_rate DOUBLE,
                test_sharpe DOUBLE,
                test_return DOUBLE,
                test_max_dd DOUBLE,
                test_num_trades INTEGER,
                test_win_rate DOUBLE,
                sharpe_degradation DOUBLE,
                is_overfit BOOLEAN,
                scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, params)
            )
        """)

        # Index for fast lookups by test sharpe
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_sharpe ON scan_results(test_sharpe DESC)
        """)

    def save_scan(self, summary: ScanSummary) -> int:
        """Save a scan summary to the database.

        Args:
            summary: Scan results to save

        Returns:
            Number of rows inserted
        """
        df = summary.to_dataframe()
        if df.is_empty():
            return 0

        # Add metadata columns
        df = df.with_columns(
            [
                pl.lit(summary.ticker).alias("ticker"),
                pl.lit(summary.train_start).alias("train_start"),
                pl.lit(summary.train_end).alias("train_end"),
                pl.lit(summary.test_start).alias("test_start"),
                pl.lit(summary.test_end).alias("test_end"),
            ]
        )

        # Convert params to JSON string
        param_cols = summary.param_names
        df = df.with_columns(
            pl.struct(param_cols)
            .map_elements(lambda x: json.dumps(dict(x)), return_dtype=pl.Utf8)
            .alias("params")
        )

        # Drop individual param columns (they're in the JSON now)
        df = df.drop(param_cols)

        # Reorder columns to match table schema (excluding id and scanned_at)
        columns = [
            "ticker",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "params",
            "train_sharpe",
            "train_return",
            "train_max_dd",
            "train_num_trades",
            "train_win_rate",
            "test_sharpe",
            "test_return",
            "test_max_dd",
            "test_num_trades",
            "test_win_rate",
            "sharpe_degradation",
            "is_overfit",
        ]
        df = df.select(columns)

        # Insert using DuckDB's Polars integration
        self.conn.execute("DELETE FROM scan_results WHERE ticker = ?", [summary.ticker])
        col_names = ", ".join(columns)
        self.conn.execute(f"INSERT INTO scan_results ({col_names}) SELECT * FROM df")

        return len(df)

    def get_all_results(self) -> pl.DataFrame:
        """Get all stored results as a DataFrame."""
        return self.conn.execute("SELECT * FROM scan_results").pl()

    def get_ticker_results(self, ticker: str) -> pl.DataFrame:
        """Get results for a specific ticker."""
        return self.conn.execute("SELECT * FROM scan_results WHERE ticker = ?", [ticker]).pl()

    def best_per_ticker(
        self,
        metric: str = "test_sharpe",
        n: int = 1,
        exclude_overfit: bool = True,
    ) -> pl.DataFrame:
        """Get best parameter combinations per ticker.

        Args:
            metric: Metric to rank by (default: test_sharpe)
            n: Number of top results per ticker
            exclude_overfit: Whether to exclude overfit results

        Returns:
            DataFrame with best results per ticker
        """
        overfit_filter = "AND is_overfit = FALSE" if exclude_overfit else ""

        query = f"""
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY {metric} DESC) as rank
                FROM scan_results
                WHERE 1=1 {overfit_filter}
            )
            SELECT * FROM ranked WHERE rank <= {n}
            ORDER BY ticker, rank
        """
        return self.conn.execute(query).pl()

    def portfolio_candidates(
        self,
        min_test_sharpe: float = 0.0,
        min_train_sharpe: float = 0.0,
        max_degradation: float = 1.0,
        exclude_overfit: bool = True,
    ) -> pl.DataFrame:
        """Get candidates suitable for portfolio construction.

        Args:
            min_test_sharpe: Minimum out-of-sample Sharpe
            min_train_sharpe: Minimum in-sample Sharpe
            max_degradation: Maximum allowed Sharpe degradation
            exclude_overfit: Whether to exclude overfit results

        Returns:
            DataFrame with qualifying candidates, one per ticker (best by test Sharpe)
        """
        overfit_filter = "AND is_overfit = FALSE" if exclude_overfit else ""

        query = f"""
            WITH filtered AS (
                SELECT *
                FROM scan_results
                WHERE test_sharpe >= {min_test_sharpe}
                  AND train_sharpe >= {min_train_sharpe}
                  AND sharpe_degradation <= {max_degradation}
                  {overfit_filter}
            ),
            best_per_ticker AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY test_sharpe DESC) as rank
                FROM filtered
            )
            SELECT * FROM best_per_ticker WHERE rank = 1
            ORDER BY test_sharpe DESC
        """
        return self.conn.execute(query).pl()

    def compare_tickers(
        self,
        tickers: list[str] | None = None,
        metric: str = "test_sharpe",
    ) -> pl.DataFrame:
        """Compare best results across tickers.

        Args:
            tickers: List of tickers to compare (default: all)
            metric: Metric to compare

        Returns:
            Pivot table with tickers as rows, metrics as columns
        """
        best = self.best_per_ticker(metric=metric, n=1, exclude_overfit=True)

        if tickers is not None:
            best = best.filter(pl.col("ticker").is_in(tickers))

        return best.select(
            [
                "ticker",
                "params",
                "train_sharpe",
                "test_sharpe",
                "train_return",
                "test_return",
                "train_max_dd",
                "test_max_dd",
                "sharpe_degradation",
            ]
        ).sort("test_sharpe", descending=True)

    def clear_ticker(self, ticker: str) -> None:
        """Remove all results for a ticker."""
        self.conn.execute("DELETE FROM scan_results WHERE ticker = ?", [ticker])

    def clear_all(self) -> None:
        """Remove all stored results."""
        self.conn.execute("DELETE FROM scan_results")

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self) -> "ResultsStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()
