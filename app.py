from omegaconf import OmegaConf
from query import VectaraQuery
import os
from PIL import Image
import uuid

import streamlit as st
from streamlit_pills import pills
from streamlit_feedback import streamlit_feedback
import plotly.express as px

from utils import thumbs_feedback, send_amplitude_data, escape_dollars_outside_latex

from dotenv import load_dotenv
load_dotenv(override=True)

max_examples = 6
languages = {'English': 'eng', 'Spanish': 'spa', 'French': 'fra', 'Chinese': 'zho', 'German': 'deu', 'Hindi': 'hin', 'Arabic': 'ara',
             'Portuguese': 'por', 'Italian': 'ita', 'Japanese': 'jpn', 'Korean': 'kor', 'Russian': 'rus', 'Turkish': 'tur', 'Persian (Farsi)': 'fas',
             'Vietnamese': 'vie', 'Thai': 'tha', 'Hebrew': 'heb', 'Dutch': 'nld', 'Indonesian': 'ind', 'Polish': 'pol', 'Ukrainian': 'ukr',
             'Romanian': 'ron', 'Swedish': 'swe', 'Czech': 'ces', 'Greek': 'ell', 'Bengali': 'ben', 'Malay (or Malaysian)': 'msa', 'Urdu': 'urd'}

# Setup for HTTP API Calls to Amplitude Analytics
if 'device_id' not in st.session_state:
    st.session_state.device_id = str(uuid.uuid4())


if "feedback_key" not in st.session_state:
        st.session_state.feedback_key = 0

def isTrue(x) -> bool:
    if isinstance(x, bool):
        return x
    return x.strip().lower() == 'true'

def plot_data(df):
    with st.expander("Data Visualization"):
        # Plot configuration
        plot_type = st.selectbox("Select Plot Type", ["Bar", "Line", "Scatter", "Box", "Histogram", "Pie"])
        
        # Get all columns for selection
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
            
            # Create plot based on selection
            if plot_type == "Bar":
                fig = px.bar(df, x=x_axis, y=y_axis)
            elif plot_type == "Line":
                fig = px.line(df, x=x_axis, y=y_axis)
            elif plot_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis)
            else:  # Box plot
                fig = px.box(df, x=x_axis, y=y_axis)
        
        # Additional customization options
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

    # left side content
    with st.sidebar:
        image = Image.open('ti-logo.png')
        st.image(image, width=350)
        # st.markdown(f"## About\n\n"
        #             f"This demo uses Vectara RAG to ask questions about {cfg.source_data_desc}\n")
        
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
        # st.markdown(
        #     "## How this works?\n"
        #     "This app was built with [Vectara](https://vectara.com).\n"
        #     "This app uses Vectara [Chat API](https://docs.vectara.com/docs/console-ui/vectara-chat-overview) to query the corpus and present the results to you, answering your question.\n\n"
        # )       

    st.markdown(f"<center> <h2> {cfg.title} </h2> </center>", unsafe_allow_html=True)

    if "messages" not in st.session_state.keys():
        reset()
                
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            if isinstance(message["content"], str):
                st.write(message["content"])
            else:  # For dataframe responses
                st.dataframe(message["content"])
                if not message["content"].empty:
                    plot_data(message["content"])

    example_container = st.empty()
    with example_container:
        if show_example_questions():
            example_container.empty()
            st.rerun()

    # select prompt from example question or user provided input
    if st.session_state.ex_prompt:
        prompt = st.session_state.ex_prompt
    else:
        prompt = st.chat_input()
        # Add pill selector for response format
        selected_format = st.pills("Response format:", ["Normal", "Table"], selection_mode="single", default="Normal")
        if prompt and selected_format == "Table":
            prompt = f"[table] {prompt}"
            
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": '🧑‍💻'})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(prompt)
        st.session_state.ex_prompt = None
        
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            if '[table]' in prompt.lower():
                with st.spinner("Generating table..."):
                    response = generate_response(prompt)
                    st.dataframe(response)
                    if not response.empty:
                        plot_data(response)
                    # Convert DataFrame to string representation for message storage
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

            # Send query and response to Amplitude Analytics
            send_amplitude_data(
                user_query=st.session_state.messages[-2]["content"],
                chat_response=str(st.session_state.messages[-1]["content"]),
                demo_name=cfg["title"],
                language=st.session_state.language
            )
            st.rerun()

# Replace the problematic condition with this:
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