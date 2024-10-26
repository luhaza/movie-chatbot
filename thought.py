"""
Please answer the following thought questions. These are questions that help you reflect
on the process of building this chatbot and about ethics. 

We are expecting at least three complete sentences for each question for full credit. 

Each question is worth 2.5 points. 
"""

######################################################################################
"""
QUESTION 1 - Anthropomorphizing

Is there potential for users of your chatbot possibly anthropomorphize 
(attribute human characteristics to an object) it? 
What are some possible ramifications of anthropomorphizing chatbot systems? 
Can you think of any ways that chatbot designers could ensure that users can easily 
distinguish the chatbot responses from those of a human?
"""

Q1_your_answer = """

I think there is potential for users of our chatbot to possibly anthropomorphize it 
because there is a possibility that after the chatbot asks the user "How are you doing today?",
the user could respond and then ask the chatbot how it is doing, anthropomorphizing it by asking
something like "How are you?". Some possible ramifications of anthropomorphizing chatbot systems
could be 1. expecting more intelligent responses out of the chatbot, and 2. becoming emotionally attached
to the chatbot (although I think this is very unlikely to occur). One way that chatbot designers
could ensure that users can easily distinguish the chatbot responses from those of a human would be to
set up the chatbot's responses such that the chatbot doesn't use pronouns to address itself, and doesn't
use adverbs like "please". Doing these things will make the chatbot's responses more robotic. 

"""

######################################################################################

"""
QUESTION 2 - Privacy Leaks

One of the potential harms for chatbots is collecting and then subsequently leaking 
(advertly or inadvertently) private information. Does your chatbot have risk of doing so? 
Can you think of ways that designers of the chatbot can help to mitigate this risk? 
"""

Q2_your_answer = """

No, our chatbot does not have any risks of collecting and then subsequently leaking private information. 
This is because our chatbot doesn't ask any private information (doesn't ask for name, age, etc), and hops
straight into movie recommendations. Designers of the chatbot can help to mitigate the risk of leaking 
private information by insuring that the private information (if used to craft more personalized responses) will 
never be added to datasets that the chatbot's features (in this case movie recommendations) are trained on. Additionally
designers can make sure that all personal/private information is not saved anywhere after a singular interaction
finishes. 

"""

######################################################################################

"""
QUESTION 3 - Classifier 

When designing your chatbot, you had the choice of using a lexicon-based sentiment analysis module 
or a statistical module (logistic regression). Which one did you end up choosing for your 
system. Why? 
"""

Q3_your_answer = """

We used the rule-based/lexicon-based sentiment analysis module. We chose this route because it performed 
better during testing of the chatbot. For a simple input like 'I like "X"', statistical predicts -1.
We felt that 'like' would be commonly used while interacting with the chatbot, so we opted for rule-based, which
predicted 1.

"""

"""
QUESTION 4 - Refelection 

You just built a frame-based dialogue system using a combination of rule-based and machine learning 
approaches. Congratulations! Reflect on the advantages and disadvantages of this paradigm. 
"""

Q4_your_answer = """

The advantages of a frame-based dialogue system using a combination of rule-based and machine learning
approaches isd that there is a clear structure, it has usecase-specific applications, it is easily modifiable/scalable,
and it uses less data than a fully machine learning approach. The clear structure makes it easy to build (just build the
individual components and combine them together in process), and the functions have clear objectives (not ambigious). Due 
to the clear structure, the program is also very scalable. We can continue to create functions for features, and it will be easy 
to incorporate those features into process. 

The disadvantages of this type of system is that it has very limited ability for open-ended discussions. Users can not have a 
back-and-forth conversation with our chatbot, compared to the extensive conversational ability of LLM chatbots like ChatGPT. There 
is also a challenge to balance the use of rule-based and machine learning approaches. We were constantly asking ourselves which type 
of approach would be better where... Finally, another disadvantage is that there is limited ability of the chatbot to perform things 
other than give movie recommendations because we would have to do an overhaul of the whole system, changing if statements, creating a 
host of new functions, etc. 
"""
