import os
from apikey import apikey

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🚀🔗 Start-up GPT Creator')
st.markdown("### Enter your keywords")
prompt = st.text_input('Enter your start-up idea or keyword(s) for your start-up:') 



# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Generate a one-word startup name reflecting {topic} and innovation.'
)


script_template = PromptTemplate(
    input_variables = ['title', 'topic'], 
    template='Craft a compelling 1-minute pitch for "{title}", a groundbreaking startup inspired by {topic}. Use analogies if useful, and convey its purpose in one line, highlighting its unique problem-solving approach.'
)


competitor_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research', 'script', 'topic'], 
    template='Analyze competitors for "{title}" startup in {topic} using Wikipedia research: {wikipedia_research}. Startup script: {script}. Provide key bullet points on potential competition and ways the startup can differentiate itself.'
)

market_research_template = PromptTemplate(
    input_variables = ['script', 'wikipedia_research', 'topic'], 
    template='Provide information on the market size, trends, and potential target audience for a start-up with this idea SCRIPT: {script} and based on this topic: {topic}, while leveraging this wikipedia research:{wikipedia_research}. Use key bullet points.'
)

revenue_model_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research', 'topic'], 
    template='Suggest revenue models to be applied to a start-up with this name TITLE: {title} and based on this topic: {topic}, while leveraging this wikipedia research:{wikipedia_research}. List in bullet points.'
)

mvp_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research', 'topic'], 
    template='Generate ideas for an MVP for a start-up with this name TITLE: {title} and based on this topic: {topic}, while leveraging this wikipedia research:{wikipedia_research}. Use bullet points.'
)

tech_stack_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research', 'topic'], 
    template='Suggest a technology stack required to develop the MVP for a start-up with this name TITLE: {title} and based on this topic: {topic}, while leveraging this wikipedia research:{wikipedia_research}. Use Bullet points.'
)

roadmap_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research', 'topic'], 
    template='Outline a step-by-step development roadmap for a start-up with this name TITLE: {title} and based on this topic: {topic}, while leveraging this wikipedia research:{wikipedia_research}.'
)



# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
competitor_memory = ConversationBufferMemory(input_key='script', memory_key='chat_history')
market_research_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
revenue_model_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
mvp_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
tech_stack_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
roadmap_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')



# Llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
competitor_chain = LLMChain(llm=llm, prompt=competitor_template, verbose=True, output_key='competitor', memory=competitor_memory)
market_research_chain = LLMChain(llm=llm, prompt=market_research_template, verbose=True, output_key='market_research', memory=market_research_memory)
revenue_model_chain = LLMChain(llm=llm, prompt=revenue_model_template, verbose=True, output_key='revenue_model', memory=revenue_model_memory)
mvp_chain = LLMChain(llm=llm, prompt=mvp_template, verbose=True, output_key='mvp', memory=mvp_memory)
tech_stack_chain = LLMChain(llm=llm, prompt=tech_stack_template, verbose=True, output_key='tech_stack', memory=tech_stack_memory)
roadmap_chain = LLMChain(llm=llm, prompt=roadmap_template, verbose=True, output_key='roadmap', memory=roadmap_memory)

wiki = WikipediaAPIWrapper()

if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research, topic=prompt)
    competitor = competitor_chain.run(title=title, wikipedia_research=wiki_research, script=script, topic=prompt)
    market_research = market_research_chain.run(title=title, script=script, wikipedia_research=wiki_research, topic=prompt)
    revenue_model = revenue_model_chain.run(title=title, wikipedia_research=wiki_research, topic=prompt)
    mvp = mvp_chain.run(title=title, wikipedia_research=wiki_research, topic=prompt)
    tech_stack = tech_stack_chain.run(title=title, wikipedia_research=wiki_research, topic=prompt)
    roadmap = roadmap_chain.run(title=title, wikipedia_research=wiki_research, topic=prompt)


    st.markdown("### Name")
    st.write(title)
    st.markdown("### 1-Minute Pitch") 
    st.write(script) 
    st.markdown("### Competitor Analysis")
    st.write(competitor)
    st.markdown("### Market Research")
    st.write(market_research)
    st.markdown("### Revenue Models")
    st.write(revenue_model)
    st.markdown("### Minimum Viable Product")
    st.write(mvp)
    st.markdown("### Tech Stack")
    st.write(tech_stack)
    st.markdown("### Roadmap")
    st.write(roadmap)

    with st.expander('Name History'): 
        st.info(title_memory.buffer)

    with st.expander('Pitch History'): 
        st.info(script_memory.buffer)

    with st.expander('Competitor Analysis'): 
        st.info(competitor_memory.buffer)

    with st.expander('Market Research'): 
        st.info(market_research_memory.buffer)

    with st.expander('Revenue Model Suggestions'): 
        st.info(revenue_model_memory.buffer)

    with st.expander('MVP Suggestions'): 
        st.info(mvp_memory.buffer)

    with st.expander('Tech Stack Recommendations'): 
        st.info(tech_stack_memory.buffer)

    with st.expander('Development Roadmap'): 
        st.info(roadmap_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)
