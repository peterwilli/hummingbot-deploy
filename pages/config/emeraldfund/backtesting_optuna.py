import asyncio
import enum
import json
import os
import random
import traceback
from datetime import datetime, timedelta
from functools import cmp_to_key
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

from frontend.pages.config.emeraldfund.utils import prepare_install
prepare_install("optuna", "optuna")
prepare_install("astor", "astor")
prepare_install("tokenize", "tokenize")
prepare_install("yapf", "yapf")

import optuna
from optuna.trial import FrozenTrial
import streamlit as st
from frontend.pages.config.emeraldfund.code_replace import code_replace
from frontend.pages.config.emeraldfund.core.utils import (
    create_slices_of_start_end_date,
)
from frontend.pages.config.emeraldfund.hyperrankrank import hyperrankrank
from frontend.st_utils import get_backend_api_client
from plotly.basedatatypes import itertools

prepare_install("streamlit-sortables", "streamlit_sortables")
from streamlit_sortables import sort_items

objective_to_name = {
    "net_pnl": "Profit",
    "largest_loss_pct": "Largest loss",
    "max_drawdown_pct": "Max Drawdown",
    "speed": "Time in seconds",
}

objective_name_to_objective = {v: k for k, v in objective_to_name.items()}


def render_trial_buttons(trials: List[List[FrozenTrial]], callback):
    cols = st.columns(len(trials))
    for trial_idx, trials in enumerate(itertools.zip_longest(*trials)):
        for col_idx, trial in enumerate(trials):
            with cols[col_idx]:
                if trial is not None:
                    st.button(
                        trial_to_string(trial, list(objective_to_name.keys())),
                        key=f"EMTrialButton_{col_idx}_{trial_idx}",
                        on_click=lambda trial=trial: callback(trial),
                    )


def render_save_best_trial_config(
    config_base_default: str, trials: List[List[FrozenTrial]]
):
    st.write("### Save the optimized config")
    backend_api_client = get_backend_api_client()
    all_configs = backend_api_client.get_all_controllers_config()
    config_bases = set(config_name["id"].split("_")[0] for config_name in all_configs)
    config_base = config_base_default.split("_")[0]
    if config_base in config_bases:
        config_tag = max(
            float(config["id"].split("_")[-1])
            for config in all_configs
            if config_base in config["id"]
        )
        version, tag = str(config_tag).split(".")
        config_tag = f"{version}.{int(tag) + 1}"
    else:
        config_tag = "0.1"
    c1, c2 = st.columns(2)
    with c1:
        config_base = st.text_input(
            "Config Base", value=config_base, key="EMTrialConfigBase"
        )
    with c2:
        config_tag = st.text_input(
            "Config Tag", value=config_tag, key="EMTrialConfigTag"
        )

    def on_upload_click(trial: FrozenTrial):
        config_data = json.loads(trial.user_attrs["config"])
        config_data["id"] = f"{config_base}_{config_tag}"
        backend_api_client.add_controller_config(config_data)
        st.session_state.EMBestTrials = None
        st.success("Config uploaded successfully!")

    render_trial_buttons(trials, on_upload_click)


def trial_to_string(trial: Optional[FrozenTrial], objectives) -> str:
    if trial is None:
        return ""
    trial_str = []
    for objective in objectives:
        trial_str.append(
            f"{objective_to_name[objective]}: {trial.user_attrs[objective]:.2f}"
        )
    return ", ".join(trial_str)


async def run_optimization_fn(
    study_name: str,
    processor,
    inputs,
    locks,
    hyperrankrank_order: List[str],
    hyperrankrank_k: int,
    objectives,
    storage_path: Path,
    amount_of_trials: int,
    date_ranges: List[List[int]],
    add_current_configuration: bool,
    backtesting_resolution: str
):
    import sys

    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(dir_path)
    sys.path.append("/backend-api/bots")
    from decimal import Decimal

    from controllers.directional_trading.directional_emeraldfund import (
        DirectionalEmeraldFundControllerConfig,
    )
    from hummingbot.strategy_v2.executors.position_executor.data_types import (
        TrailingStop,
    )

    from core.backtesting.optimizer import (
        BacktestingConfig,
        BaseStrategyConfigGenerator,
        StrategyOptimizer,
    )

    class EmeraldFundConfigGenerator(BaseStrategyConfigGenerator):
        def __init__(self, date_ranges, processor, inputs, locks):
            super().__init__(date_ranges)
            self.processor = processor
            self.inputs = inputs
            self.locks = locks

        def get_current_trial_params(self) -> Dict[str, Any]:
            result = {}
            inputs = self.inputs
            processor = self.processor
            parameters = processor.get_parameters()
            for k in parameters:
                param = parameters[k]
                result[k] = param["current"]
            result.update(inputs)
            result["trailing_stop_activation_price"] = inputs["trailing_stop"][
                "activation_price"
            ]
            result["trailing_stop_trailing_delta"] = inputs["trailing_stop"][
                "trailing_delta"
            ]
            return result

        def check_lock(self, k, x, fn):
            if self.locks[k]:
                return x
            else:
                return fn()

        def generate_config(self, trial) -> BacktestingConfig:
            # TODO: Make sure PMM config works too
            inputs = self.inputs
            processor = self.processor
            parameters = processor.get_parameters()
            for k in parameters:
                param = parameters[k]
                kt = type(param["current"])
                if kt is int:
                    param_max = param["max"]
                    param_min = param["min"]
                    param["current"] = trial.suggest_int(k, param_min, param_max)
                elif kt is float:
                    param_max = param["max"]
                    param_min = param["min"]
                    step = None
                    if "step" in param:
                        step = param["step"]
                    param["current"] = trial.suggest_float(
                        k, param_min, param_max, step=step
                    )
                elif kt is str:
                    param["current"] = trial.suggest_categorical(k, param["choices"])
                else:
                    raise ValueError(f"unknown type {kt}")

            processor_code = code_replace(inputs["processor_code"], parameters)
            max_executors_per_side = self.check_lock(
                "max_executors_per_side",
                inputs["max_executors_per_side"],
                lambda: trial.suggest_int("max_executors_per_side", 1, 10),
            )
            take_profit = self.check_lock(
                "take_profit",
                inputs["take_profit"],
                lambda: trial.suggest_float("take_profit", 0.01, 0.5, step=0.01),
            )
            stop_loss = self.check_lock(
                "stop_loss",
                inputs["stop_loss"],
                lambda: trial.suggest_float("stop_loss", 0.01, 0.1, step=0.01),
            )
            trailing_stop_activation_price = self.check_lock(
                "trailing_stop_activation_price",
                inputs["trailing_stop"]["activation_price"],
                lambda: trial.suggest_float(
                    "trailing_stop_activation_price", 0.005, 0.05, step=0.001
                ),
            )
            trailing_stop_trailing_delta = self.check_lock(
                "trailing_stop_trailing_delta",
                inputs["trailing_stop"]["trailing_delta"],
                lambda: trial.suggest_float(
                    "trailing_stop_trailing_delta",
                    max(0.001, trailing_stop_activation_price * 0.01),
                    max(0.01, trailing_stop_activation_price * 0.9),
                ),
            )
            time_limit = self.check_lock(
                "time_limit",
                inputs["time_limit"],
                lambda: trial.suggest_int("time_limit", 60, 60 * 60 * 24, step=60),
            )
            cooldown_time = self.check_lock(
                "cooldown_time",
                inputs["cooldown_time"],
                lambda: trial.suggest_int("cooldown_time", 60, 60 * 60 * 12, step=60),
            )
            interval = self.check_lock(
                "interval",
                inputs["interval"],
                lambda: trial.suggest_categorical(
                    "interval", ["1m", "3m", "5m", "15m", "1h"]
                ),
            )

            # Create the strategy configuration
            config = DirectionalEmeraldFundControllerConfig(
                processor_code=processor_code,
                connector_name=inputs["connector_name"],
                trading_pair=inputs["trading_pair"],
                candles_connector=inputs["candles_connector"],
                candles_trading_pair=inputs["candles_trading_pair"],
                interval=interval,
                max_records=inputs["max_records"],
                total_amount_quote=Decimal(inputs["total_amount_quote"]),
                take_profit=Decimal(take_profit),
                stop_loss=Decimal(stop_loss),
                trailing_stop=TrailingStop(
                    activation_price=Decimal(trailing_stop_activation_price),
                    trailing_delta=Decimal(trailing_stop_trailing_delta),
                ),
                time_limit=time_limit,
                max_executors_per_side=max_executors_per_side,
                cooldown_time=cooldown_time,
            )

            # Return the configuration encapsulated in BacktestingConfig
            return BacktestingConfig(
                config=config, resolution=backtesting_resolution, date_ranges=date_ranges
            )

    config_generator = EmeraldFundConfigGenerator(
        date_ranges=date_ranges, processor=processor, inputs=inputs, locks=locks
    )

    sqlite_path = storage_path / Path("studies.db")
    optimizer = StrategyOptimizer(
        storage_name=f"sqlite:///{sqlite_path}", objectives=objectives
    )
    study = optimizer._create_study(study_name, load_if_exists=True)
    study_bar = st.progress(0, "Running study...")
    trial_status = st.empty()
    trial_status_header = (
        " | "
        + " | ".join(
            map(
                lambda x: f"Rank {x[0] + 1}: {objective_to_name[x[1]]}",
                enumerate(hyperrankrank_order),
            )
        )
        + " | \n"
    )
    trial_status_header += " | " + (" | ----- |" * len(hyperrankrank_order)) + " | "
    if add_current_configuration:
        study.enqueue_trial(config_generator.get_current_trial_params())

    sorters = {
        "max_drawdown_pct": cmp_to_key(
            lambda x, y: y.user_attrs["max_drawdown_pct"]
            - x.user_attrs["max_drawdown_pct"]
        ),
        "largest_loss_pct": cmp_to_key(
            lambda x, y: y.user_attrs["largest_loss_pct"] - x.user_attrs["largest_loss_pct"]
        ),
        "speed": cmp_to_key(lambda x, y: x.user_attrs["speed"] - y.user_attrs["speed"]),
        "net_pnl": cmp_to_key(
            lambda x, y: y.user_attrs["net_pnl"] - x.user_attrs["net_pnl"]
        ),
    }

    hyperrankrank_sorters = list(map(lambda x: sorters[x], hyperrankrank_order))
    best_trials = None
    trial_status_table = None
    early_stop = False

    def early_stop_fn():
        nonlocal early_stop
        early_stop = True
        st.session_state.EMBestTrials = {
            "best_trials": best_trials,
        }

    st.button("Stop Study", key="EMStopStudyBtn", on_click=early_stop_fn)

    for i in range(amount_of_trials):
        if early_stop:
            break

        trial = study.ask()
        try:
            # Run the async objective function and get the result
            value = await optimizer._async_objective(trial, config_generator)
            # Report the result back to the study
            study.tell(trial, value)
        except Exception:
            print(f"Error in study: {traceback.format_exc()}")  # noqa: F821
            study.tell(trial, state=optuna.trial.TrialState.FAIL)

        study_bar.progress(
            i / amount_of_trials,
            f"Running study {i + 1} / {amount_of_trials}...",
        )
        best_trials = hyperrankrank(
            list(
                filter(
                    lambda x: all([k in x.user_attrs for k in hyperrankrank_order]),
                    study.get_trials(deepcopy=False),
                )
            ),
            hyperrankrank_k,
            hyperrankrank_sorters,
            breakdown=True,
        )
        trial_status_rows = []
        for trials in itertools.zip_longest(*best_trials):
            string_trials = list(
                map(lambda x: trial_to_string(x, hyperrankrank_order), trials)
            )
            trial_status_rows.append(" | " + " | ".join(string_trials) + " | ")
        trial_status_table = trial_status_header + "\n" + "\n".join(trial_status_rows)
        trial_status.write(trial_status_table)

    study_bar.progress(1.0, "Done!")
    trial_status.write("")
    st.session_state.EMBestTrials = {
        "best_trials": best_trials,
    }


def optuna_section(inputs, backend_api_client, processor):
    st.write("### Optimization Backtesting")
    c1, c2, c3 = st.columns(3)

    with c1:
        study_name = st.text_input("Study name")
    with c2:
        backtesting_resolution = st.selectbox(
            "Backtesting Resolution",
            options=["1m", "3m", "5m", "15m", "30m", "1h", "1s"],
            index=0,
            key="optuna_backtesting_resolution",
        )
    with c3:
        trade_cost = st.number_input(
            "Trade Cost (%)",
            min_value=0.0,
            value=0.06,
            step=0.01,
            format="%.2f",
            key="optuna_trade_cost",
        )

    default_end_time = datetime.now().date() - timedelta(days=1)
    default_start_time = default_end_time - timedelta(days=2)
    c1, c2, c3 = st.columns(3)
    with c1:
        start_date = st.date_input(
            "Start Date", value=default_start_time, key="EMOptunaStartTime"
        )

    with c2:
        end_date = st.date_input(
            "End Date", value=default_end_time, key="EMOptunaEndTime"
        )

    break_up_sections = st.checkbox(
        "Break up start and end date into random sections",
        key="EMOptunaBreakUpDate",
        value=False,
    )

    if break_up_sections:
        c1, c2, c3 = st.columns(3)
        with c1:
            amount_of_sections = st.number_input("Amount of sections", value=3)
        with c2:
            break_up_gap = st.number_input(
                "Gap in days between sections", key="EMOptunaBreakUpGap", value=0
            )
        seed = st.session_state.get("EMRngSeed", random.randint(0, 2**32))
        if "EMRngSeed" not in st.session_state:
            st.session_state.EMRngSeed = seed
        with c3:
            jitter = st.number_input("Jitter", key="EMOptunaJitter", value=0)
        sections = create_slices_of_start_end_date(
            int(datetime.combine(start_date, datetime.min.time()).timestamp()),
            int(datetime.combine(end_date, datetime.min.time()).timestamp()),
            amount_of_sections,
            break_up_gap,
            jitter,
            seed=seed,
        )
        sections_table = dedent("""
        | Section # | Start Date | End Date | Days | Total Days |
        | ------------- | ------------- | ------------- | ------------- | ---------- |
        """)

        def display_unix_timestamp(ts: int) -> str:
            return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

        total_days = 0
        for idx, section in enumerate(sections):
            start_date = display_unix_timestamp(section[0])
            end_date = display_unix_timestamp(section[1])
            days = (section[1] - section[0]) // 86400
            total_days += days
            sections_table += (
                f"{idx + 1} | {start_date} | {end_date} | {days} | {total_days}\n"
            )

        st.write(sections_table)
        st.write("\n")
    else:
        sections = [
            [
                int(datetime.combine(start_date, datetime.min.time()).timestamp()),
                int(datetime.combine(end_date, datetime.min.time()).timestamp()),
            ]
        ]

    with c3:
        amount_of_trials = st.number_input(
            "Amount of Trials", value=1000, min_value=1, key="EMOptunaAmountOfTrials"
        )

    with st.expander("Objectives 📈"):
        st.write(
            "Objectives give you a custom direction to which your strategy is to be optimized."
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            objective_net_pnl = st.checkbox(
                objective_to_name["net_pnl"], value=True, key="EMOptunaObjectiveNetPNL"
            )
        with c2:
            objective_largest_loss = st.checkbox(
                objective_to_name["largest_loss_pct"], value=True, key="EMOptunaObjectiveLargestLoss"
            )
        with c3:
            objective_max_drawdown = st.checkbox(
                objective_to_name["max_drawdown_pct"],
                value=True,
                key="EMOptunaObjectiveMaxDrawdown",
            )
        with c4:
            objective_speed = st.checkbox(
                objective_to_name["speed"], value=False, key="EMOptunaObjectiveSpeed"
            )

        objectives = []
        if objective_net_pnl:
            objectives.append("net_pnl")
        if objective_largest_loss:
            objectives.append("largest_loss_pct")
        if objective_max_drawdown:
            objectives.append("max_drawdown_pct")
        if objective_speed:
            objectives.append("speed")
        st.write("**Objective importance order**")
        st.write("Move to the left what is most important for you")

        hyperrankrank_order = sort_items(list(objective_to_name.values()))
        hyperrankrank_order = list(
            map(lambda x: objective_name_to_objective[x], hyperrankrank_order)
        )
        hyperrankrank_k = st.number_input(
            "Rank size (Higher means more variation)", value=3, key="EMRankSize"
        )
    locks = {}
    with st.expander("Locks 🔒"):
        st.write(
            "Locks will make sure that selected variables will not change during optimization. If you have any preferences such as a certain risk management that you dont want to change, this is where you can do it!"
        )

        lock_all_btn = st.button(
            "Lock all",
            key="EMLockAllBtn",
        )
        if lock_all_btn:
            st.session_state.EMTrailingStopActPriceLock = True
            st.session_state.EMTrailingStopDeltaLock = True
            st.session_state.EMTakeProfitLock = True
            st.session_state.EMStopLossLock = True
            st.session_state.EMMaxExecutorsPerSideLock = True
            st.session_state.EMIntervalLock = True
            st.session_state.EMTimeLimitLock = True
            st.session_state.EMCooldownTimeLock = True

        c1, c2, c3 = st.columns(3)
        with c1:
            locks["trailing_stop_activation_price"] = st.checkbox(
                "Trailing Stop Act. Price",
                key="EMTrailingStopActPriceLock",
            )
            locks["trailing_stop_trailing_delta"] = st.checkbox(
                "Trailing Stop Delta",
                key="EMTrailingStopDeltaLock",
            )
        with c2:
            locks["take_profit"] = st.checkbox("Take Profit", key="EMTakeProfitLock")
            locks["stop_loss"] = st.checkbox("Stop Loss", key="EMStopLossLock")
        with c3:
            locks["max_executors_per_side"] = st.checkbox(
                "Max Executors Per Side",
                key="EMMaxExecutorsPerSideLock",
            )
            locks["interval"] = st.checkbox("Interval", key="EMIntervalLock")
            locks["time_limit"] = st.checkbox("Time Limit", key="EMTimeLimitLock")
            locks["cooldown_time"] = st.checkbox(
                "Cooldown Time", key="EMCooldownTimeLock"
            )

    add_current_configuration = st.checkbox(
        "Add current configuration to study",
        key="EMOptunaAddCurrentConfiguration",
        value=False,
    )
    run_optimization = st.button("Run Optimization")
    if run_optimization:
        if len(study_name) == 0:
            raise ValueError("Study name is required")
        st.session_state.EMBestTrials = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        storage_path = Path(dir_path) / Path("data")
        storage_path.mkdir(parents=True, exist_ok=True)
        asyncio.run(
            run_optimization_fn(
                study_name,
                processor,
                inputs,
                locks,
                hyperrankrank_order,
                hyperrankrank_k,
                objectives,
                storage_path,
                int(amount_of_trials),
                sections,
                add_current_configuration,
                backtesting_resolution
            )
        )
    if st.session_state.get("EMBestTrials", None) is not None:
        render_save_best_trial_config(
            st.session_state["default_config"]["id"],
            st.session_state.EMBestTrials["best_trials"],
        )
