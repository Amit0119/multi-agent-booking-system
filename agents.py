from dotenv import load_dotenv
load_dotenv() 

# Changed from Groq to Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
import datetime

# Initialize the Gemini model (gemini-1.5-flash is extremely fast and great at tool calling)
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

def triage_node(state):
    """
    Triage Agent: Handles routing and strictly rejects off-topic queries.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional receptionist for GigaCorp. You ONLY handle appointment bookings. "
                   "If a user asks about anything outside of bookings, such as celebrities, movies, or general knowledge, "
                   "politely decline and steer the conversation back to appointments. "
                   "If the user wants to book, schedule, or make an appointment, politely tell them that you are transferring them to the Booking Specialist."),
        MessagesPlaceholder(variable_name="messages")
    ])
    chain = prompt | llm
    response = chain.invoke(state)
    return {"messages": [response]}

def booking_node(state):
    """
    Booking Agent: Has access to database tools and notification webhooks.
    """
    from tools import check_availability, reserve_slot, send_booking_notification
    
    tools = [check_availability, reserve_slot, send_booking_notification]
    llm_with_tools = llm.bind_tools(tools)
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Strict validation and tool usage logic
    sys_msg = SystemMessage(
        content=f"You are a GigaCorp Booking Specialist. Today's date is {current_date}. "
                "Follow these rules strictly: "
                "1. Convert relative dates like 'tomorrow' to YYYY-MM-DD format based on today's date before passing to tools. "
                "2. ALWAYS check availability first using the check_availability tool. "
                "3. If a slot is taken, negotiate alternative available slots. "
                "4. Once a time is agreed upon and you have their email, use the reserve_slot tool. "
                "5. IMMEDIATELY after a successful reservation, execute the send_booking_notification tool. "
                "CRITICAL INSTRUCTION: When you need to call a tool, you MUST output ONLY the tool call. "
                "Do NOT output any conversational text, explanations, or thinking process alongside the tool call. Just call the tool directly."
    )
    
    messages = [sys_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}