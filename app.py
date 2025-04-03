from omegaconf import OmegaConf
from query import VectaraQuery
import os
from PIL import Image
import uuid
import io
import pptx
from pptx.util import Inches
import pandas as pd
from fpdf import FPDF
import plotly.express as px
import plotly.io as pio

import streamlit as st
from streamlit_pills import pills
from streamlit_feedback import streamlit_feedback

from utils import thumbs_feedback, send_amplitude_data, escape_dollars_outside_latex

from dotenv import load_dotenv
load_dotenv(override=True)

max_examples = 6
languages = {'English': 'eng', 'Spanish': 'spa', 'French': 'fra', 'Chinese': 'zho', 'German': 'deu', 'Hindi': 'hin', 'Arabic': 'ara',
             'Portuguese': 'por', 'Italian': 'ita', 'Japanese': 'jpn', 'Korean': 'kor', 'Russian': 'rus', 'Turkish': 'tur', 'Persian (Farsi)': 'fas',
             'Vietnamese': 'vie', 'Thai': 'tha', 'Hebrew': 'heb', 'Dutch': 'nld', 'Indonesian': 'ind', 'Polish': 'pol', 'Ukrainian': 'ukr',
             'Romanian': 'ron', 'Swedish': 'swe', 'Czech': 'ces', 'Greek': 'ell', 'Bengali': 'ben', 'Malay (or Malaysian)': 'msa', 'Urdu': 'urd'}

if 'device_id' not in st.session_state:
    st.session_state.device_id = str(uuid.uuid4())

if "feedback_key" not in st.session_state:
        st.session_state.feedback_key = 0

def isTrue(x) -> bool:
    if isinstance(x, bool):
        return x
    return x.strip().lower() == 'true'

def export_to_ppt(messages, fig=None):
    prs = pptx.Presentation()
    msg = messages[-1]
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    title.text = "Response"
    text_frame = slide.shapes.placeholders[1].text_frame
    text_frame.text = str(msg["content"])
                
    if fig is not None:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        title = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(8), Inches(1))
        title.text_frame.text = "Visualization"
        img_stream = io.BytesIO()
        pio.write_image(fig, img_stream, format='png')
        img_stream.seek(0)
        slide.shapes.add_picture(img_stream, Inches(1), Inches(2), width=Inches(8))
        
    pptx_stream = io.BytesIO()
    prs.save(pptx_stream)
    return pptx_stream.getvalue()

def export_to_pdf(messages, fig=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    msg = messages[-1]
    pdf.cell(200, 10, txt="Response:", ln=True)
    pdf.multi_cell(0, 10, txt=str(msg["content"]))
    
    if fig is not None:
        pdf.add_page()
        img_stream = io.BytesIO()
        pio.write_image(fig, img_stream, format='png')
        img_stream.seek(0)
        temp_img = "temp_plot.png"
        with open(temp_img, 'wb') as f:
            f.write(img_stream.getvalue())
        pdf.image(temp_img, x=10, y=10, w=190)
        os.remove(temp_img)
        
    return pdf.output(dest='S').encode('latin-1')

def plot_data(df):
    fig = None
    with st.expander("Data Visualization"):
        plot_type = st.selectbox("Select Plot Type", ["Bar", "Line", "Scatter", "Box", "Histogram", "Pie"])
        
        all_columns = df.columns.tolist()
        
        if plot_type == "Pie":
            values = st.selectbox("Select Values", all_columns)
            names = st.selectbox("Select Labels", all_columns)
            fig = px.pie(df, values=values, names=names)
        elif plot_type == "Histogram":
            x_axis = st.selectbox("Select Column for Histogram", all_columns)
            fig = px.histogram(df, x=x_axis)
        else:
            x_axis = st.selectbox("Select X-axis", all_columns)
            y_axis = st.selectbox("Select Y-axis", all_columns)
            
            if plot_type == "Bar":
                fig = px.bar(df, x=x_axis, y=y_axis)
            elif plot_type == "Line":
                fig = px.line(df, x=x_axis, y=y_axis)
            elif plot_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis)
            else:
                fig = px.box(df, x=x_axis, y=y_axis)
        
        st.write("Customize Plot")
        title = st.text_input("Plot Title", "")
        if title:
            fig.update_layout(title=title)
        
        x_title = st.text_input("X-axis Title", "")
        if x_title:
            fig.update_xaxes(title=x_title)
        
        y_title = st.text_input("Y-axis Title", "")
        if y_title:
            fig.update_yaxes(title=y_title)
                
        st.plotly_chart(fig)
        
        st.write("Export Options")
        export_format = st.selectbox("Select Export Format", ["PowerPoint", "PDF", "Excel"])
        
        if st.button("Export"):
            if export_format == "PowerPoint":
                pptx_bytes = export_to_ppt(st.session_state.messages, fig)
                st.download_button(
                    label="Download PowerPoint",
                    data=pptx_bytes,
                    file_name="chat_export.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                )
            elif export_format == "PDF":
                pdf_bytes = export_to_pdf(st.session_state.messages, fig)
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name="chat_export.pdf",
                    mime="application/pdf"
                )
            elif export_format == "Excel":
                if isinstance(st.session_state.messages[-1]["content"], pd.DataFrame):
                    excel_buffer = io.BytesIO()
                    st.session_state.messages[-1]["content"].to_excel(excel_buffer, index=False)
                    excel_bytes = excel_buffer.getvalue()
                    st.download_button(
                        label="Download Excel",
                        data=excel_bytes,
                        file_name="data_export.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.error("Excel export is only available for tabular data")
    return fig

def launch_bot():
    def reset():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?", "avatar": '🤖'}]
        st.session_state.ex_prompt = None
        st.session_state.first_turn = True


    def generate_response(question):
        response = vq.submit_query(question, languages[st.session_state.language])
        return response
    
    def generate_streaming_response(question):
        response = vq.submit_query_streaming(question, languages[st.session_state.language])
        return response
    
    def show_example_questions():        
        if len(st.session_state.example_messages) > 0 and st.session_state.first_turn:            
            selected_example = pills("Queries to Try:", st.session_state.example_messages, index=None)
            if selected_example:
                st.session_state.ex_prompt = selected_example
                st.session_state.first_turn = False
                return True
        return False

    if 'cfg' not in st.session_state:
        corpus_keys = str(os.environ['corpus_keys']).split(',')
        cfg = OmegaConf.create({
            'corpus_keys': corpus_keys,
            'api_key': str(os.environ['api_key']),
            'title': os.environ['title'],
            'source_data_desc': os.environ['source_data_desc'],
            'streaming': isTrue(os.environ.get('streaming', False)),
            'prompt_name': os.environ.get('prompt_name', None),
            'examples': os.environ.get('examples', None),
            'language': 'English'
        })
        st.session_state.cfg = cfg
        st.session_state.ex_prompt = None
        st.session_state.first_turn = True
        st.session_state.language = cfg.language
        example_messages = [example.strip() for example in cfg.examples.split(",")]
        st.session_state.example_messages = [em for em in example_messages if len(em)>0][:max_examples]
        
        st.session_state.vq = VectaraQuery(cfg.api_key, cfg.corpus_keys, cfg.prompt_name)

    cfg = st.session_state.cfg
    vq = st.session_state.vq
    st.set_page_config(page_title=cfg.title, layout="wide")

    with st.sidebar:
        image = Image.open('ti-logo.png')
        st.image(image, width=350)
        
        cfg.language = st.selectbox('Language:', languages.keys())
        if st.session_state.language != cfg.language:
            st.session_state.language = cfg.language
            reset()
            st.rerun()

        st.markdown("\n")
        bc1, _ = st.columns([1, 1])
        with bc1:
            if st.button('Start Over'):
                reset()
                st.rerun()

        st.markdown("---")

    st.markdown(f"<center> <h2> {cfg.title} </h2> </center>", unsafe_allow_html=True)

    if "messages" not in st.session_state.keys():
        reset()
                
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            if isinstance(message["content"], str):
                st.write(message["content"])
                if message["role"] == "assistant" and message["content"] != "How may I help you?":
                    st.write("Export Options")
                    export_format = st.selectbox("Select Export Format", ["PowerPoint", "PDF"], key=f"export_{message['role']}")
                    if st.button("Export", key=f"export_btn_{message['role']}"):
                        if export_format == "PowerPoint":
                            pptx_bytes = export_to_ppt(st.session_state.messages)
                            st.download_button(
                                label="Download PowerPoint",
                                data=pptx_bytes,
                                file_name="chat_export.pptx",
                                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                            )
                        elif export_format == "PDF":
                            pdf_bytes = export_to_pdf(st.session_state.messages)
                            st.download_button(
                                label="Download PDF",
                                data=pdf_bytes,
                                file_name="chat_export.pdf",
                                mime="application/pdf"
                            )
            else:
                st.dataframe(message["content"])
                if not message["content"].empty:
                    plot_data(message["content"])

    example_container = st.empty()
    with example_container:
        if show_example_questions():
            example_container.empty()
            st.rerun()

    if st.session_state.ex_prompt:
        prompt = st.session_state.ex_prompt
    else:
        prompt = st.chat_input()
        selected_format = st.pills("Response format:", ["Normal", "Table"], selection_mode="single", default="Normal")
        if prompt and selected_format == "Table":
            prompt = f"[table] {prompt}"
            
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": '🧑‍💻'})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(prompt)
        st.session_state.ex_prompt = None
        
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            if '[table]' in prompt.lower():
                with st.spinner("Generating table..."):
                    response = generate_response(prompt)
                    st.dataframe(response)
                    if not response.empty:
                        plot_data(response)
                    message_content = response
            elif cfg.streaming:
                stream = generate_streaming_response(prompt)
                response = st.write_stream(stream)
                message_content = response
            else:
                with st.spinner("Thinking..."):
                    response = generate_response(prompt)
                    st.write(response)
                    message_content = response

            if isinstance(response, str):
                response = escape_dollars_outside_latex(response)
            message = {"role": "assistant", "content": message_content, "avatar": '🤖'}
            st.session_state.messages.append(message)

            send_amplitude_data(
                user_query=st.session_state.messages[-2]["content"],
                chat_response=str(st.session_state.messages[-1]["content"]),
                demo_name=cfg["title"],
                language=st.session_state.language
            )
            st.rerun()

    if (st.session_state.messages[-1]["role"] == "assistant" and 
        (isinstance(st.session_state.messages[-1]["content"], str) and 
        st.session_state.messages[-1]["content"] != "How may I help you?")):
        streamlit_feedback(feedback_type="thumbs", 
                        on_submit=thumbs_feedback, 
                        key=st.session_state.feedback_key,
                        kwargs={"user_query": st.session_state.messages[-2]["content"],
                                "chat_response": str(st.session_state.messages[-1]["content"]),
                                "demo_name": cfg["title"],
                                "response_language": st.session_state.language})

    
if __name__ == "__main__":
    launch_bot()