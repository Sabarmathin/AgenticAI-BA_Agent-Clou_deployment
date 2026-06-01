import streamlit as st
import autogen
from autogen import AssistantAgent
import os
import google.generativeai as genai
# 1. Page Configuration & UI Layout
st.set_page_config(page_title="BA Agentic Pipeline", page_icon="", layout="wide")
st.title("AI Business Analyst Pipeline")
st.subheader("Extract features, generate user stories, and create vetted acceptance criteria.")

# 2. Sidebar Configuration for Credentials
with st.sidebar:
    st.header("Configuration")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("API Key not found! Please add GOOGLE_API_KEY to your .env file.")
        st.stop()    
    genai.configure(api_key=api_key)
    gemini_key = api_key
    st.markdown("---")
    st.caption("Powered by AutoGen & Gemini 3.1 Flash Lite")

# 3. Text Area for Raw Requirements Ingestion
raw_requirements = st.text_area(
    "Paste your raw requirement document text here:", 
    height=250, 
    placeholder="Example: The application must allow users to log in with Google OAuth..."
)

# 4. Process Workflow Execution on Button Click
if st.button("Process Requirements", type="primary"):
    if not gemini_key:
        st.error("Please provide your Gemini API key in the sidebar to proceed.")
    elif not raw_requirements.strip():
        st.warning("Please paste some text requirements first!")
    else:
        # Define LLM configuration inside the trigger to use the live user-provided key
        llm_config = {
            "config_list": [
                {
                    "model": "gemini-3.1-flash-lite",
                    "api_key": gemini_key,
                    "api_type": "google"
                }
            ],
            "temperature": 0.2,
        }
        #BA Manager Agent to provide the BRD document and control the flow
        BA_Manager_Agent = autogen.AssistantAgent(
            name="BA_Manager_Agent",
            human_input_mode="NEVER",
            is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
            llm_config=llm_config,
            system_message="You are a BA Manager. You assign the task to each agent to read the requirement document."
                        " Extract the features in the requirement document "
                        "Create user stories based on the extracted features "
                        "Write the acceptance criteria for each user stories."
                        "Finally you review the acceptance criteria inline with the requirements",
        )
        
        # Extract features from BRD
        feature_extract_agent = autogen.AssistantAgent(
            name="feature_extract_agent",
            system_message="You are a Feature extract agent. Extract distinct, atomic features from raw requirements documents BRD.pdf. " 
                "Expert at parsing messy text and identifying core business logic.",
            llm_config=llm_config,
        )
        
        # Convert Features into user stories
        User_story_agent = autogen.AssistantAgent(
            name="User_story_agent",
            llm_config=llm_config,
            system_message="You are User story creation agent "
                "Create the user stories based on the extracted features from the BRD document,"
                "Master of the standard format: 'As a [user], I want [action] so that [benefit]'."
                "Ensure all the requirements mentioned in the BRD are covered in the user stories"
                "concrete and to the point."
                "Begin the creation by stating your role.",
        )
        
        # Create Acceptance Criteria for each user stories
        AC_agent = autogen.AssistantAgent(
            name="AC_agent",
            llm_config=llm_config,
            system_message="You are a Acceptance Criteria Agent "
                "You are ability to create acceptance criteria for each user story"
                "For each story, provide 2 detailed Acceptance Criteria using Given-When-Then syntax"
                "You ensure that each point is covered in the user story and inline with the BRD document."
                "concrete and to the point. "
                "Begin the creation by stating your role.",
        )
        
        # Nested Chat to solve the task
        def reflection_message(recipient, messages, sender):
            return f'''Review the following content. 
                    \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}'''
        
        BA_chats = [
            # {
            #  "recipient": ragproxyagent, 
            #  "message": reflection_message, 
            #  "summary_method": "reflection_with_llm",
            #  # "summary_args": {"summary_prompt" : 
            #  #    "Return features into as JSON object only:"
            #  #    "{'Agent': '', 'Features': ''}. Here Agent should be your role",},
            #  "max_turns": 1},
            {
             "recipient": feature_extract_agent, 
             "message": reflection_message, 
             "summary_method": "reflection_with_llm",
             "summary_args": {"summary_prompt" : 
                "Return features into as JSON object only:"
                "{'Agent': '', 'Features': ''}. Here Agent should be your role",},
             "max_turns": 1},
            {
            "recipient": User_story_agent, 
            "message": reflection_message, 
             "summary_method": "reflection_with_llm",
             "summary_args": {"summary_prompt" : 
                "Return User stories into as JSON object only:"
                "{'Agent': '', 'User Stories': ''}.Here Agent should be your role",},
             "max_turns": 1},
            {"recipient": AC_agent, 
             "message": reflection_message, 
             "summary_method": "reflection_with_llm",
             "summary_args": {"summary_prompt" : 
                "Return Acceptance Criteria for each story into as JSON object only:"
                "{'Agent': '', 'AC': ''}.Here Agent should be your role",},
             "max_turns": 1},     
        ]
        # Running the agentic system with Streamlit loading indicators
        with st.status("🧠 Agents are analyzing requirements... Please wait.", expanded=True) as status:
            st.write("🏃‍♂️ Dispatching Business Analyst agent to map features...")
            chat_results = BA_Manager_Agent.initiate_chats(BA_chats)
            status.update(label="✅ Pipeline Completed!", state="complete", expanded=False)

        # 5. Display the structured final output nicely in the Web UI
        st.success("🎉 Requirements Processing Complete!")
        
        st.markdown("### 📋 Final Vetted Backlog Output")
        # Grabs the summary text from the final step in the chat pipeline
        st.markdown(chat_results[-1].summary)
