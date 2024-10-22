import streamlit as st
import os

main_page = st.Page(
    page = "views\main.py",
    title = "Invoice Extractor Project",
    default = True
)

pdf_invoice_page = st.Page(
    page = "views\pdf_extractor.py",
    title = "PDF Invoice Information extractor",
    default = False
)
jpg_invoice_page = st.Page(
    page = "views\jpg_invoice_app.py",
    title = "Image Invoice Information extractor",
    default = False
)


pg = st.navigation(
    {
        "Main Menu":[main_page],
        "Invoice Extraction formats":[pdf_invoice_page ,jpg_invoice_page]
    })
st.sidebar.text("Made by Saurav R for Zolvit")
pg.run()