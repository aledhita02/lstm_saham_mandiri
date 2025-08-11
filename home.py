import streamlit as st

pages = {
    "Menu": [
        st.Page("pages/dataset.py", title="Dataset",
                icon=":material/dataset:"),
        st.Page("pages/prediksi.py", title="Prediksi Close Price",
                icon=":material/analytics:"),
        # st.Page("pages/evaluasi.py", title="Evaluasi",
        #         icon=":material/speed:"),
    ],
}

pg = st.navigation(pages)
pg.run()
