import nltk
from nltk.chat.util import Chat, reflections




# Define the patterns and responses for the chatbot
patterns = [
    #hello
    (r"Hi|Hey|Is anyone there?|Hello|Hay",["Hello", "Hi", "Hi there"]),
    (r"Hey there whats up|whats up|what's up|hey there what's up",["Not much, just hanging out in cyberspace "]),
    (r"Thanks|Thank you|That's helpful|Thanks for the help",["Happy to help!", "Any time!", "My pleasure", "You're most welcome!"]),
    (r"Who are you?|What are you?|Who you are?",["I.m simple", "hiyour bot assistant", "I'm simple, an Artificial Intelligent bot"]),
    (r"what is your name|what should I call you|whats your name?",["You can call me Simple.", "I'm Simple!", "Just call me as Simple"]),
    (r"my name is (.*)", ["Hello %1, how can I help you today?"]),
    
    (r"have a complaint|I want to raise a complaint|there is a complaint about a service",["Please provide us your complaint in order to assist you", "Please mention your complaint, we will reach you and sorry for any inconvenience caused"]),
    (r"I need to create a new account|how to open a new account|I want to create an account|can you create an account for me|how to open a new account",["You can just easily create a new account from our web site", "Just go to our web site and follow the guidelines to create a new account"]),
    (r"Could you help me?|give me a hand please|Can you help?|What can you do for me?|I need a support|I need a help|support me please",["Tell me how can assist you", "Tell me your problem to assist you", "Yes Sure, How can I support you"]),
    (r"what can you do|what are your capabilities", ["I can answer questions about a variety of topics."]),
    (r"what is your name|who are you", ["My name is Chatbot. Nice to meet you!"]),

    (r"How do I contact customer support?|how can i connext support|support|Customer care|Customer contact|contact",["You can contact our customer support team via email at support@company.com","You can contact our customer support team via phone at 1-800-555-1234.. Is there anything else you need help with?"]),
    (r"I need help|help|doute|I have a doute",["Ya i here to help you ..:) can you specify the thing"]),

    #exit
    (r"quit|exit", ["Goodbye!"]),
    (r"Bye|See you later|Goodbye",["See you later", "Have a nice day", "Bye! Come back again"]),
    (r"Thank you for your help|thanks for your help|thank you so much|well done|Good work",
     ["You're welcome! Don't hesitate to reach out if you have any further questions or if you need assistance with anything else. Have a great day!"]),

    #informal
    (r"i love you|i love|i love u|you look cute|love you",["i love you too...( ◡‿◡ *)","tnx..!👉👈","love you more🫰"]),
    (r"i like you|like you",["i like you too","tnx"]),
    (r"i like to eat some thing|i want to eat some thing|i want eat|im hungry",[" i suggest maggi it won'what take more that 2 min hahaha...! "]),
    #self
    (r"Tell me about your self|yourself|tell me about you|who are you",["this is a AI bot"]),

]

# Create a chatbot object and initialize it with the patterns
chatbot = Chat(patterns, reflections)
print("hi...! this is a Silmple an AI bot")

print()
print("how can i help you")


# Start the chatbot

print ("chat:",chatbot.converse())
