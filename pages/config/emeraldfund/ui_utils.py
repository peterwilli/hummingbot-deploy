import streamlit as st
from frontend.pages.config.emeraldfund.utils import get_examples


def render_examples(example_type, inputs):
    with st.expander("Examples", expanded=False):
        examples = get_examples(example_type)
        current_example = st.selectbox(
            "Select an example", examples, format_func=lambda x: x[0]
        )
        st.code(current_example[1], language="python")
        do_import = st.button(label="Import")
        if do_import:
            inputs["processor_code"] = current_example[1]
