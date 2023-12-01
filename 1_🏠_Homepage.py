import streamlit as st
import subprocess

st.set_page_config(
    page_title="Home"
)

st.header("Text analyzer")
st.write("""
            This application allows you to perform various types of text analysis.
         
            Types of text analysis that you can perform right now:
         
            - Sentiment analysis - input some text and see what an emotional tone it has: 🙂, 😐 or ☹️.
         
                Current supported languages: Russian. 
            """)

print(subprocess.run(["/workspaces/nlp-text-analysis/model_load.sh"]))
