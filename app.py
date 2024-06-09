from langchain_cohere import ChatCohere
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable.config import RunnableConfig
from langchain.agents import AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from takehome.tools import query_vector_db_articles, get_smiles_from_pubchem, predict_energy_from_smiles

import chainlit as cl

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="What disease are they discussing in the paper 'Targeted therapy in advanced non-small cell lung cancer: current advances and future trends.'?",
            message="What disease are they discussing in the paper 'Targeted therapy in advanced non-small cell lung cancer: current advances and future trends.'? ",
            icon="/public/PDF_icon.svg",
            ),

        cl.Starter(
            label="Find SMILES of Molecule",
            message="Can you find the SMILES string of Ibuprofen?",
            icon="/public/Caffeine.svg",
            ),
        cl.Starter(
            label="Predict Hydration Free Energy",
            message="Can you predict the energy value of 'CCO'? ",
            icon="/public/lightning.svg",
            ),

        ]

@cl.on_chat_start
async def on_chat_start():
    model = ChatCohere(streaming=True)
    prompt = ChatPromptTemplate.from_template("{input}")
    preamble = """
    You are an expert who answers the user's question with the most relevant datasource.
    You must answer in a precise and concise manner.
    You are equipped with a vectordatabase about scientific articles,
    a tool to get the SMILES string of a given molecule name,
    and a tool to predict the energy of a molecule from its SMILES string.
    """

    # Create the ReAct agent
    agent = create_cohere_react_agent(
    llm=model,
    tools=[query_vector_db_articles,get_smiles_from_pubchem, predict_energy_from_smiles],
    prompt=prompt,
    )

    agent_executor = AgentExecutor(agent=agent,
                                tools=[query_vector_db_articles,get_smiles_from_pubchem, predict_energy_from_smiles],
                                verbose=True,
                                handle_parsing_errors=True,)
    
    memory = ChatMessageHistory()

    runnable = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: memory,
        memory=memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

 
    cl.user_session.set("runnable", runnable)
    cl.user_session.set("preamble", preamble)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    preamble = cl.user_session.get("preamble")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    config = RunnableConfig(callbacks=[cb],configurable= {"session_id": "<foo>"})
    result = runnable.invoke({"input":message.content,
                              "preamble": preamble}, config=config)

    await cl.Message(content=result['output']).send()