import pandas_ta as ta  # noqa: F401
from frontend.visualization import theme
import plotly.graph_objects as go
from pathlib import Path


def clean_consecutive_values(s, steps=1):
    for idx in range(steps):
        s = s.where(s.ne(s.shift(idx + 1)), 0)
    return s


def get_examples(example_type: str) -> list[tuple[str, str]]:
    """
    Returns a list of tuples containing the file name and contents of each file in the specified example_type directory.

    :param example_type: A string representing the type of examples, which is also the directory name.
    :return: A list of tuples (file_name, contents) for each file in the directory.
    """
    examples = []
    example_path = Path(__file__).parent / Path("examples") / Path(example_type)

    if not example_path.exists() or not example_path.is_dir():
        raise ValueError(
            f"The provided example_type '{example_type}' does not exist or is not a directory ({example_path})."
        )

    for file in example_path.iterdir():
        if file.is_file():
            with file.open("r", encoding="utf-8") as f:
                contents = f.read()
                examples.append((file.name, contents))

    return examples


def get_market_making_traces(df, price_multiplier, spread_multiplier):
    tech_colors = theme.get_color_scheme()
    upper_bound = (df["close"] * (1 + price_multiplier)) * (1 + spread_multiplier)
    lower_bound = (df["close"] * (1 + price_multiplier)) * (1 - spread_multiplier)
    traces = [
        go.Scatter(
            x=df.index,
            y=upper_bound,
            line=dict(color=tech_colors["upper_band"]),
            name="Upper Band",
        ),
        go.Scatter(
            x=df.index,
            y=lower_bound,
            line=dict(color=tech_colors["lower_band"]),
            name="Lower Band",
        ),
    ]
    return traces


def prepare_install(package_name: str, module_name: str):
    import importlib.util

    # Check if the module exists
    if importlib.util.find_spec(module_name) is None:
        # Install the module using pip
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

        # Reload the module
        import importlib

        importlib.reload(importlib.util)
