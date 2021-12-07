import os
import streamlit.components.v1 as components

_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "st_card",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("st_card", path=build_dir)

def st_card(title, value, unit='', delta=None, delta_description=None, use_percentage_delta=False, show_progress=False, progress_value=False, key=None):
    if show_progress:
        if progress_value:
            component_value = _component_func(title=title, value=value, unit=unit, delta=None, show_progress=show_progress, progress_value=progress_value, key=key)
        else:
            component_value = _component_func(title=title, value=value, unit=unit, delta=None, show_progress=show_progress, progress_value=value, key=key)
    else:
        component_value = _component_func(title=title, value=value, unit=unit, delta=delta, delta_description=delta_description, use_percentage_delta=use_percentage_delta, show_progress=show_progress, key=key)
    return component_value

if __name__ == "__main__":
    import streamlit as st
    st.title('Streamlit Card Test')
    cols = st.columns([.333,.333,.333])
    with cols[0]:
        st_card('Orders', value=1200, delta=-45, delta_description='since last month')
    with cols[1]:
        st_card('Competed Orders', value=76.4, unit='%', show_progress=True)
    with cols[2]:
        st_card('Profit', value=45000, unit='($)', delta=48, use_percentage_delta=True, delta_description='since last year')