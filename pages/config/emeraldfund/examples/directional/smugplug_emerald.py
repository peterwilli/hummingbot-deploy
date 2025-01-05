"""
Smugplug Strategy + Emerald's modifications

Additions:

    - Allows for fine-tuning the ATR, making it less active in sideways markets, the primary weakness of this strategy
"""


class SignalProcessor:
    def get_parameters(self):
        return {
            "atr_threshold_lookback": {"current": 50, "max": 50, "min": 1},
            "atr_threshold_compare_lookback": {"current": 10, "max": 10, "min": 1},
            "atr_threshold_multiplier": {
                "current": 0.95,
                "max": 1.0,
                "min": 0.5,
                "step": 0.01,
            },
            "atr_period": {"current": 10, "max": 10, "min": 1},
            "macd_fast": {"current": 1, "max": 50, "min": 1},
            "macd_signal": {"current": 41, "max": 50, "min": 5},
            "macd_slow": {"current": 5, "max": 50, "min": 1},
            "macd_hist_lookback": {"current": 2, "max": 10, "min": 1},
            "long_ema_period": {"current": 43, "max": 50, "min": 1},
            "medium_ema_period": {"current": 18, "max": 50, "min": 1},
            "short_ema_period": {"current": 31, "max": 50, "min": 5},
        }

    def decimate(self, series):
        start_of_sequence = series & ~series.shift(fill_value=False)
        group_number = start_of_sequence.cumsum()
        result = series & (
            group_number
            == group_number.where(series).groupby(group_number).transform("first")
        )
        result = series & ~series.shift(fill_value=False)
        return result

    def process_candles(self, candles: pd.DataFrame) -> pd.DataFrame:
        macd_fast = self.macd_fast
        macd_slow = self.macd_fast + self.macd_slow
        macd_signal = self.macd_signal
        short_ema_period = self.short_ema_period
        medium_ema_period = self.medium_ema_period + self.short_ema_period
        long_ema_period = (
            self.long_ema_period + self.medium_ema_period + self.short_ema_period
        )
        candles["short_ema"] = ta.ema(candles["close"], length=short_ema_period)
        candles["medium_ema"] = ta.ema(candles["close"], length=medium_ema_period)
        candles["long_ema"] = ta.ema(candles["close"], length=long_ema_period)
        candles["atr"] = ta.atr(
            candles["high"], candles["low"], candles["close"], length=self.atr_period
        )
        candles["rolling_atr_max"] = (
            candles["atr"]
            .rolling(window=self.atr_threshold_lookback)
            .max()
            .shift(self.atr_threshold_compare_lookback)
            * self.atr_threshold_multiplier
        )
        voilatility_ok = candles["atr"] > candles["rolling_atr_max"]
        macd = ta.macd(
            candles["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal
        )
        candles["macd"] = macd[f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"]
        candles["macd_signal"] = macd[f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"]
        candles["macd_hist"] = macd[f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"]
        candles["signal"] = 0
        buy_condition = (
            (candles["short_ema"] > candles["medium_ema"])
            & (candles["medium_ema"] > candles["long_ema"])
            & (candles["close"] > candles["short_ema"])
            & voilatility_ok
            & (candles["macd"] > candles["macd_signal"])
            & (candles["macd_hist"] > 0)
            & (
                candles["macd_hist"]
                > candles["macd_hist"].shift(self.macd_hist_lookback)
            )
        )
        buy_condition = self.decimate(buy_condition)
        candles.loc[buy_condition, "signal"] = 1
        sell_condition = (
            (candles["short_ema"] < candles["medium_ema"])
            & (candles["medium_ema"] < candles["long_ema"])
            & (candles["close"] < candles["short_ema"])
            & voilatility_ok
            & (candles["macd"] < candles["macd_signal"])
            & (candles["macd_hist"] < 0)
            & (
                candles["macd_hist"]
                < candles["macd_hist"].shift(self.macd_hist_lookback)
            )
        )
        sell_condition = self.decimate(sell_condition)
        candles.loc[sell_condition, "signal"] = -1
        # candles['line_separate_macd'] = candles['macd']
        # candles['line_separate_macd_signal'] = candles['macd_signal']
        # candles['line_separate_macd_hist'] = candles['macd_hist']
        # candles['line_separate_short_ema'] = candles['short_ema']
        # candles['line_separate_medium_ema'] = candles['medium_ema']
        # candles['line_separate_long_ema'] = candles['long_ema']
        candles["line_separate_atr"] = candles["atr"]
        candles["line_separate_atr_threshold"] = candles["rolling_atr_max"]
        return candles
