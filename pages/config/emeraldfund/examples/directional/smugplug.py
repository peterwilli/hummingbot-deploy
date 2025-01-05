"""
The Smugplug strategy is a technical trading strategy that combines multiple indicators
to generate buy and sell signals. It primarily uses Exponential Moving Averages (EMAs)
to determine the trend direction, Average True Range (ATR) to set price thresholds, and
Moving Average Convergence Divergence (MACD) to confirm momentum. The strategy aims to
capture both trend-following and momentum-based opportunities.

Trend Direction:
    - The strategy uses three EMAs with different periods (short, medium, and long)
      to identify the trend direction.
    - A bullish trend is confirmed when the short EMA is above the medium EMA, and
      the medium EMA is above the long EMA.
    - Conversely, a bearish trend is confirmed when the short EMA is below the medium EMA,
      and the medium EMA is below the long EMA.

Price Thresholds:
    - The ATR is used to set dynamic price thresholds for entry and exit.
    - For buy signals, the closing price must be above the short EMA and above the
      closing price of the previous candle minus the ATR multiplied by a multiplier.
    - For sell signals, the closing price must be below the short EMA and below the
      closing price of the previous candle plus the ATR multiplied by a multiplier.

Momentum Confirmation:
    - The MACD and its histogram are used to confirm the momentum of the trend.
    - A buy signal is generated when the MACD line is above the MACD signal line,
      the MACD histogram is positive, and the histogram is increasing.
    - A sell signal is generated when the MACD line is below the MACD signal line,
      the MACD histogram is negative, and the histogram is decreasing.

The strategy is designed to be robust by incorporating multiple indicators and dynamic
price thresholds, aiming to minimize false signals and improve trade quality.
"""


class SignalProcessor:
    def get_parameters(self):
        return {
            "atr_multiplier": {"current": 1.5, "max": 5, "min": 1},
            "atr_period": {"current": 11, "max": 30, "min": 5},
            "long_ema_period": {"current": 31, "max": 100, "min": 30},
            "macd_fast": {"current": 21, "max": 26, "min": 8},
            "macd_signal": {"current": 9, "max": 20, "min": 5},
            "macd_slow": {"current": 42, "max": 60, "min": 20},
            "medium_ema_period": {"current": 29, "max": 50, "min": 15},
            "short_ema_period": {"current": 8, "max": 20, "min": 5},
        }

    def process_candles(self, candles: pd.DataFrame) -> pd.DataFrame:
        # Check if required columns exist
        required_columns = ["close", "high", "low"]
        if not all(col in candles.columns for col in required_columns):
            raise ValueError(
                f"Input dataframe must contain columns: {required_columns}"
            )

        # Calculate EMAs
        candles["short_ema"] = ta.ema(candles["close"], length=self.short_ema_period)
        candles["medium_ema"] = ta.ema(candles["close"], length=self.medium_ema_period)
        candles["long_ema"] = ta.ema(candles["close"], length=self.long_ema_period)

        # Calculate ATR
        candles["atr"] = ta.atr(
            candles["high"], candles["low"], candles["close"], length=self.atr_period
        )

        # Calculate MACD
        macd = ta.macd(
            candles["close"],
            fast=self.macd_fast,
            slow=self.macd_slow,
            signal=self.macd_signal,
        )
        candles["macd"] = macd[
            f"MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
        ]
        candles["macd_signal"] = macd[
            f"MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
        ]
        candles["macd_hist"] = macd[
            f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
        ]

        # Generate signals
        candles["signal"] = 0

        # Get previous signal
        candles["prev_signal"] = candles["signal"].shift(1).fillna(0)

        # Adjusted base buy conditions without ADX
        base_buy_condition = (
            (candles["short_ema"] > candles["medium_ema"])
            & (candles["medium_ema"] > candles["long_ema"])
            & (candles["close"] > candles["short_ema"])
            & (
                candles["close"]
                > (candles["close"].shift(1) - self.atr_multiplier * candles["atr"])
            )
            & (candles["macd"] > candles["macd_signal"])
            & (candles["macd_hist"] > 0)
            & (
                candles["macd_hist"] > candles["macd_hist"].shift(1)
            )  # MACD histogram increasing
        )

        # Adjusted buy signals without trend confirmation
        buy_condition = base_buy_condition
        candles.loc[buy_condition, "signal"] = 1

        # Adjusted base sell conditions without ADX
        base_sell_condition = (
            (candles["short_ema"] < candles["medium_ema"])
            & (candles["medium_ema"] < candles["long_ema"])
            & (candles["close"] < candles["short_ema"])
            & (
                candles["close"]
                < (candles["close"].shift(1) + self.atr_multiplier * candles["atr"])
            )
            & (candles["macd"] < candles["macd_signal"])
            & (candles["macd_hist"] < 0)
            & (
                candles["macd_hist"] < candles["macd_hist"].shift(1)
            )  # MACD histogram decreasing
        )

        # Adjusted sell signals without trend confirmation
        sell_condition = base_sell_condition
        candles.loc[sell_condition, "signal"] = -1

        return candles
