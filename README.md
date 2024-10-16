#  Midterm Project: Movie Chatbot

The goals of this homework is to build a dialogue system that makes movie recommendations to a user. You will use a frame-based approach and implement both rule-based and machine-learning based modules.  


## Submission instructions

- This assignment will be done in **groups of 2 students**. 
- Your submitted solution and code must be your group's work alone. Please remember the 50-foot rule. Copying and pasting a solution from the internet or another source is considered a violation of the **honor code**.
- Because this is a group submission, there are no late days and no late submissions will be accepted. 
- Submit one submission per group to gradescope. To submit run the following command on the command line from this folder: 
	
	```
	./generate_submission.sh 
	```
	Then submit `submission.zip` to Gradescope. [Here are instructions](https://guides.gradescope.com/hc/en-us/articles/21863861823373-Adding-Group-Members-to-a-Submission) for adding group members to a Gradescope submission.

## Goal with example 

The goal of your chatbot is to provide movie recommendations to a user. In order to do so, the chatbot will need to:
 
1. Extract and transform a user's input 
2. Predict the sentiment of a user's input about movies 
3. Gather 5 sentiment ratings from the user to predict a new movie 

Here's an example of what an interaction between your chatbot and a human user might look like. You goal is to emulate this functionality (but chatbot's exact wording can change). 

```
chatbot> Hi! I'm MovieBot! I'm going to recommend a movie to you. First I will ask you about your taste in (at least) five movies.  
Tell me about a movie that you've seen recently (with the title in quotation marks).

user> I really liked Moneyball.
```
The bot gets confused because it only recognizes movies in quotation marks. 

```
chatbot> Sorry, I don't understand. Tell me about a *single* movie that you've seen with the title in quotation marks.

user> I like "Moneyball".

chatbot> You liked "Moneyball". Thank you! I have your preferences for 1/5 movies. Tell me about another movie you have seen.

user> I was completely amazed at how "Rush Hour" was pretty much the most ridiculous thing I've seen in my entire life.
``` 
The sentence above was too hard for the bot to predict sentiment, but the bot recovers. 

```
chatbot > I'm sorry, I'm not quite sure if you liked "Rush Hour". Tell me more about "Rush Hour" (possibly in simpler terms).

user> I hate "Rush Hour".

chatbot> Ok. You disliked "Rush Hour." There are multiple Rush Hours. 
Did you mean "Rush Hour (1998)", "Rush Hour 2 (2001)", or "Rush Hour 3 (2007)"?

user> Rush Hour 2

chatbot> Ok. You meant you disliked "Rush Hour 2 (2001)." Thank you! I have your preferences for 2/5 movies. Tell me about another movie you have seen.
```
Above, the chatbot is able to disambiguate between multiple movies that match "Rush Hour" in its movie database. 

```
user> I like love "Copycat"
```
... the bot proceeds to gather sentiment ratings from the user for 5 movies. 

```
chatbot> Thanks! I have five movies to make a recommendation. 
I suggest you watch "Happy Gilmore (1996)". Would you like to hear another recommendation? (Yes/No)
(Or enter :quit if you're done.)

user> Yes 

chatbot> I suggest you watch "Kicking and Screaming (1995)". Would you like to hear another recommendation (Yes/No)? (Or enter :quit if you're done.)

user> :quit

chatbot> Thanks! It was fun chatting with you!  

```
Your goal is to write the code that will interact with users in this manner and beyond. 

## REPL 

Dialogue systems can be decomposed into four primary modules ([Norvig 1991](https://github.com/norvig/paip-lisp)) which are called the Read-Eval-Print-Loop (REPL): 

1. Read the input.
2. Extract relevant information from the input, which can be domain specific – as in our movie recommender chatbot.
3. Transform the input into a response – users normally expect a response in the given domain.
4. Print the response.

In `repl.py` we have implemented 1 and 4 for you. In `chatbot.py` you'll implement 2 and 3. 

## Getting started

To get started: 

1. Activate `cs375` by typing on the command line 
	
	```
	conda activate cs375
	```

1. In this directory, type the following command to start the chatbot 
 
	```
	python repl.py 
	```

1. You can test your chatbot with `.txt` files containing the user input as follows: 

	```
	python repl.py < testing/simple.txt
	```
	
	As you can see, each line of `simple.txt` is entered consecutively into the chatbot. However, the script terminates once it hits `:quit` and any lines afterwards will not be executed.
	
	Files like `simple.txt` are useful when you want to test the same script multiple times and we encourage you to write your own. We will be manully testing your code with similar (but not the same) scripts as the ones provided in `testing/`.

4. Now it's your turn. Open the `chatbot.py` file and follow the instructions in the function docstrings. You can see more about the grading breakdown below. 

	
## Grading

Your grade will consist of your work for the following functions in `chatbot.py` and `thought.py`. 

| Part | Functions to implement | Points | Auto-graded?|
| -------- | -------- | -------- | -------- |
| 1 - Warm-up    | `intro()`, `greeting()`, and `goodbye()`  | 5 | No |
| 2 - Extracting and Transforming | `extract_titles()` | 5 | Yes |
| | `find_movies_idx_by_title()` | 5 | Yes |
| | `disambiguate_candidates()` | 5 | Yes | 
| | ||**Checkpoint (Parts 1&2): Due Th Oct 17 10pm** |
| 3 - Sentiment | `predict_sentiment_rule_based()` | 5 | Yes|
| | `train_logreg_sentiment_classifier()` | 10 | Yes | 
| | `predict_sentiment_statistical()` | 4 | Yes | 
| 4 - Recommend |`recommend_movies()` | 2 | Yes|
| 5 - Control flow | `process()` | 20 | No - see below |
| 6 - Open-ended | `function1()` | 10 | No - see below | 
| | `function2()` | 10 | No - see below | 
| | Additional functions | Extra credit | No | 
| 7 - Thought questions | Response in `thought.py` | 10 | No| 
||||**Full assigment due: Th Oct 24 10pm**||

By Thursday Oct 17, you must submit code for Parts 1 & 2 to the Gradescope autograder. A failure to do so by the checkpoint date will result in a loss of those autograder points for the final submission. 

### Grading `process()`

We will be manully testing your code with similar (but not the same) scripts as the ones provided in `testing/`.

You'll have to choose whether your implementation of `process()` uses the machine-learning or rule-based approach to sentiment that you implement. There are advantages and disadvantages to both. 

Make sure your bot:

- Communicates to the user the extracted sentiment and movie 
- Speaks reasonably fluently
- Fail gracefully 
	- Bot never crashes 
	- Asks for clarification
- Gives recommendations (asks automatically after user provides 5 data points).
The bot should give one recommendation at a time, each time giving the user a chance to say "yes" to hear another or "no" to finish.


### Grading Part 5 - Open Ended

For `function1()`, you must choose from the following options: 

1. Identify movies without quotation marks and incorrect capitalization
	- Example: Users type `'I liked 10 things i HATE about you'`
1. Identify and respond to emotions 
	- Example: Users type `'I am angry'`
1. Extract sentiment for multiple movies in a single user input 
	- Examples: Users type `'I didn't like either "I, Robot" or "Ex Machina".'` or `'I liked "Titanic (1997)", but "Ex Machina" was not good.'`
1. Deal with simple spelling mistakes
	- Example: User types `'I liek "Avatar"'` and the bot is able to correct `"liek"` to `"like"` 
1. Deal with articles in movie titles
	- Example: User types  `'I liked "An American in Paris"'` and your chatbot knows this matches `'American in Paris, An (1951)'` from the movie databse.
1. Use the movie categories from `data/movies.txt`, e.g. "Action" or "Thriller", as part of the input and recommendation to the user.  


For `function2()`, you may implement an additional one of the choices listed above or come up with a functionality (of equal implementation difficulty) that your group believes enhances the chatbot. Please document the functionality in the docstring so that we can evaluate and grade it properly. 

**Extra credit.** If you implement any functions beyond two, these will count towards extra credit. Feel free to go beyond functionality of just movie recommendations. You can put any additional data or code you need beyond the starter code in `deps/` folder and this will be zipped with your submission when you run `generate_submission.sh`. 


Similar to previous assignments: Extra credit can only help you and will not hurt you. At the end of the semester, if you have a borderline grade, your grade could be increased given your efforts on extra credit.

## Data

We have provided the following three datasets which you will use in `chatbot.py`. 

### Sentiment lexicon 

In `data/sentiment.txt`, we provide a sentiment lexicon that you can use to extract sentiment from the input. It consists of 3624 words with their associated sentiment (positive or negative) extracted from Harvard Inquirer (Stone et al. 1966). The lexicon is stored for your convenience in a dictionary/hash map, where the word is the key and the sentiment the value.

### Rotten Tomatoes Reviews

In `data/rotten_tomatoes.pkl`, we provide a subset of the [Rotten Tomatoes dataset from Kaggle](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/data). This dataset includes reviews in the form of text and class labels "fresh" and "rotten". 


### Movie recommendation database 

Your `movie database` consits of two files: 

- `data/movies.txt` 
- `data/ratings.txt`

This data comes from [MovieLens](https://movielens.org/) and consists of a total of 9125 movies rated by 671 users. Feel free to browse this data in a text editor. 

The file `data/ratings.txt` includes a `9125 x 671` utility matrix that contains ratings for users and movies. The ratings range anywhere from 1.0 to 5.0 with increments of 0.5. The code will binarize the ratings as follows:

```
+1 if the user liked the movie (3.0-5.0)
-1 if the user didn’t like the movie (0.5-2.5)
0 if the user did not rate the movie
```

We also provide `data/movies.txt`, a list with 9125 movie titles and their associated movie genres. The movie in the first position in the list corresponds to the movie in the first row of the ratings matrix. An example entry looks like:

```
['Blade Runner (1982)', 'Action|Sci-Fi|Thriller']
```


##  Tips for developing, testing and debugging 

### Collaborating

Since you will be developing this code in a group, we highly recommend you use a private repository on `Github` and add each other as collaborators. Use version control and `git` commands (`commit`, `push`, `pull`) to share code. 

### Developing

We *highly* suggest you open the `scratch.ipynb` (provided in this repository) in a Jupyter Notebook and use it to help speed-up development. This is an easy way to check the outputs of chunks of code before you put them into functions.

### Debug mode 
The debug method in the `Chatbot()` class can be used to print out debug information about the internal state of the chatbot that you may consider relevant, in a way that is easy to understand. This is helpful while developing the assignment, but will not be graded. To enable debug info, type in the chatbot session in the REPL

```
:debug on
```
and type

```
:debug off
```
to disable it. 


### Testing

Try how your chatbot does with the scripts in `testing/`. Recall, you can test these via 

```
python repl.py < testing/simple.txt
```

Write some of your own user conversations that you want your bot to do well on and store these in `testing/`.






