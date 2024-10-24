"""
Class that implements the chatbot for CSCI 375's Midterm Project. 

Please follow the TODOs below with your code. 
"""
import numpy as np
import argparse
import joblib
import re  
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
import nltk 
from collections import defaultdict, Counter
from typing import List, Dict, Union, Tuple

import util

class Chatbot:
    """Class that implements the chatbot for CSCI 375's Midterm Project"""

    def __init__(self):
        self.name = 'Letterbot'

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, self.ratings = util.load_ratings('data/ratings.txt')
        
        # Load sentiment words 
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        # Train the classifier
        self.train_logreg_sentiment_classifier()

        self.user_movie_ratings = {}

        self.emotion_flag = 1
        self.need_disambiguation = []
        self.d_flag = 0
        self.recent_sentiment = None
        self.can_make_rec = len(self.user_movie_ratings) >= 5

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def intro(self) -> str:
        """
        Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """

        """
        Your task is to implement the chatbot as detailed in the
        instructions (README.md).

        To exit: write ":quit" (or press Ctrl-C to force the exit)

        TODO: Write the description for your own chatbot here in the `intro()` function.
        """
        intro_message = "This chatbot will provide you with movie recommendations.\nPlease put all movie titles in double quotes (\"\")\nType :quit to quit"

        return intro_message

    def greeting(self) -> str:
        """Return a message that the chatbot uses to greet the user."""

        # greeting_message = f'Hi! I am {self.name}! I am a chatbot that specializes in movie recommendations. My personal favorite is "The Prestige." What is yours?'
        greeting_message = f'Hi! I am {self.name}, a chatbot that specializes in movie recommendations. How are you doing today?'

        return greeting_message

    def goodbye(self) -> str:
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
       
        goodbye_message = "I hope my recommendations helped. Have a nice day!"

        return goodbye_message

    def debug(self, line):
        """
        Returns debug information as a string for the line string from the REPL

        No need to modify this function. 
        """
        return str(line)

    ############################################################################
    # 2. Extracting and transforming                                           #
    ############################################################################

    def process(self, line: str) -> str:
        """
        Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this script.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'
        
        Arguments: 
            - line (str): a user-supplied line of text
        
        Returns: a string containing the chatbot's response to the user input

        Hints: 
            - We recommend doing this function last (after you've completed the
            helper functions below)
            - Try sketching a control-flow diagram (on paper) before you begin 
            this 
            - We highly recommend making use of the class structure and class 
            variables when dealing with the REPL loop. 
            - Feel free to make as many helper funtions as you would like 
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################

        response = ""

        if self.d_flag:
            # self.d_flag = 0
            d = self.disambiguate_candidates(line, self.need_disambiguation[-1])
            if len(d) == 1:
                self.user_movie_ratings[d[0]] = self.recent_sentiment
                print(self.user_movie_ratings)
                self.d_flag = 0
                self.need_disambiguation.pop()
                response += "Thank you!\n"
            elif len(d) == 0:
                response += "Clarification did not help. Please try again."
                return response
            else:
                response += "I'm sorry. I still do not understand. Did you mean:"
                for index in d:
                    response += f"\n- {self.titles[index][0]}"
                return response

        if len(self.need_disambiguation) > 0:
            response += "Which did you mean:"
            for index in self.need_disambiguation[-1]:
                response += f"\n- {self.titles[index][0]}"
            self.d_flag = 1
            return response


        spell_checked = self.function1(line)
        if spell_checked != line:
            response += f"I'm assuming you meant {spell_checked}."

        if self.emotion_flag:
            self.emotion_flag = 0
            # return self.function2(line)
            return "Sorry to hear that."


        safe_fail_response = "I did not pick up any movie titles in your response. Make sure to wrap each title in double quotes."
        movie_indices = {}

        # extract titles from input
        movie_titles = self.clean_articles(self.extract_titles(line))
        input_cardinality = len(movie_titles)

        if input_cardinality == 0 and "Thank you" not in response:
            return safe_fail_response
        elif "Thank you" in response:
            response += "What other movies do you have strong opinions on? I need at least 5 opinions to make a recommendation."
            return response

        # analyze sentiment
        self.recent_sentiment = self.predict_sentiment_rule_based(line)

        # get indices
        for movie in movie_titles:
            movie_indices[movie] = self.find_movies_idx_by_title(movie)

            if len(movie_indices[movie]) > 1:
                self.need_disambiguation.append(movie_indices[movie])
            else:
                self.user_movie_ratings[movie_indices[movie][0]] = self.recent_sentiment
                print(self.user_movie_ratings)

        if len(self.need_disambiguation) > 0:
            response += " There are multiple movies with names similar to one or more of your inputs. Say anything to proceed with disambiguation."
            return response
        

        # make recommendation if possible + user wants to
        # if self.can_make_rec:
        #     ""
                        
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    def extract_titles(self, user_input: str) -> List[str]:
        """
        Extract potential movie titles from the user input.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example 1:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I do not like any movies'))
          print(potential_titles) // prints []

        Example 2:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        Example 3: 
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'There are "Two" different "Movies" here'))
          print(potential_titles) // prints ["Two", "Movies"]                              
    
        Arguments:     
            - user_input (str) : a user-supplied line of text

        Returns: 
            - (list) movie titles that are potentially in the text

        Hints: 
            - What regular expressions would be helpful here? 
        """
        regex = r'"(.*?)"'
        matches = re.findall(regex, user_input)                                      
        return matches

    def find_movies_idx_by_title(self, title:str) -> List[int]:
        """ 
        Given a movie title, return a list of indices of matching movies
        The indices correspond to those in data/movies.txt.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list that contains the index of that matching movie.

        Example 1:
          ids = chatbot.find_movies_idx_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        Example 2:
          ids = chatbot.find_movies_idx_by_title('Twelve Monkeys')
          print(ids) // prints [31]

        Arguments:
            - title (str): the movie title 

        Returns: 
            - a list of indices of matching movies

        Hints: 
            - You should use self.titles somewhere in this function.
              It might be helpful to explore self.titles in scratch.ipynb
            - You might find one or more of the following helpful: 
              re.search, re.findall, re.match, re.escape, re.compile
            - Our solution only takes about 7 lines. If you're using much more 
            than that try to think of a more concise approach 
        """
        ret = []

        title = re.escape(title)
        regex = rf'{title}'

        for index, t in enumerate(self.titles):
            if re.search(regex, t[0], re.IGNORECASE):
                ret.append(index)

        return ret

    def disambiguate_candidates(self, clarification: str, candidates: list) -> List[int]: 
        """
        Given a list of candidate movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (e.g. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If the clarification does not uniquely identify one of the movies, this 
        should return multiple elements in the list which the clarification could 
        be referring to. 

        Example 1 :
          chatbot.disambiguate_candidates("1997", [1359, 2716]) // should return [1359]

          Used in the middle of this sample dialogue 
              moviebot> 'Tell me one movie you liked.'
              user> '"Titanic"''
              moviebot> 'Which movie did you mean:  "Titanic (1997)" or "Titanic (1953)"?'
              user> "1997"
              movieboth> 'Ok. You meant "Titanic (1997)"'

        Example 2 :
          chatbot.disambiguate_candidates("1994", [274, 275, 276]) // should return [274, 276]

          Used in the middle of this sample dialogue
              moviebot> 'Tell me one movie you liked.'
              user> '"Three Colors"''
              moviebot> 'Which movie did you mean:  "Three Colors: Red (Trois couleurs: Rouge) (1994)"
                 or "Three Colors: Blue (Trois couleurs: Bleu) (1993)" 
                 or "Three Colors: White (Trzy kolory: Bialy) (1994)"?'
              user> "1994"
              movieboth> 'I'm sorry, I still don't understand.
                            Did you mean "Three Colors: Red (Trois couleurs: Rouge) (1994)" or
                            "Three Colors: White (Trzy kolory: Bialy) (1994)" '
    
        Arguments: 
            - clarification (str): user input intended to disambiguate between the given movies
            - candidates (list) : a list of movie indices

        Returns: 
            - a list of indices corresponding to the movies identified by the clarification

        Hints: 
            - You should use self.titles somewhere in this function 
            - You might find one or more of the following helpful: 
              re.search, re.findall, re.match, re.escape, re.compile
        """
        matches = []

        clarification = re.escape(clarification.strip())

        for index in candidates:
            title = self.titles[index][0]

            if re.search(clarification, title, re.IGNORECASE):
                matches.append(index)

        return matches
        

    ############################################################################
    # 3. Sentiment                                                             #
    ########################################################################### 

    def predict_sentiment_rule_based(self, user_input: str) -> int:
        """
        Predict the sentiment class given a user_input

        In this function you will use a simple rule-based approach to 
        predict sentiment. 

        Use the sentiment words from data/sentiment.txt which we have already 
        loaded for you in self.sentiment. 
        
        Then count the number of tokens that are in the positive sentiment category 
        (pos_tok_count) and negative sentiment category (neg_tok_count).

        This function should return 
        -1 (negative sentiment): if neg_tok_count > pos_tok_count
        0 (neural): if neg_tok_count is equal to pos_tok_count
        +1 (postive sentiment): if neg_tok_count < pos_tok_count

        Example:
          sentiment = chatbot.predict_sentiment_rule_based('I LOVE "The Titanic"'))
          print(sentiment) // prints 1
        
        Arguments: 
            - user_input (str) : a user-supplied line of text
        Returns: 
            - (int) a numerical value (-1, 0 or 1) for the sentiment of the text

        Hints: 
            - Take a look at self.sentiment (e.g., in scratch.ipynb)
            - Remember we want the count of *tokens* not *types*
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################      

        regex = r'\b\w+\b'
        tokens = re.findall(regex, user_input.lower())
        pos_tok_count, neg_tok_count = 0,0

        for token in tokens:
            if token in self.sentiment:
                token_sentiment = self.sentiment[token]
                if token_sentiment == "pos":
                    pos_tok_count += 1
                else:
                    neg_tok_count += 1
        
        if neg_tok_count > pos_tok_count:
            return -1
        elif neg_tok_count < pos_tok_count:
            return 1 
        else:
            return 0
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def train_logreg_sentiment_classifier(self):
        """
        Trains a bag-of-words Logistic Regression classifier on the Rotten Tomatoes dataset 

        You'll have to transform the class labels (y) such that: 
            -1 inputed into sklearn corresponds to "rotten" in the dataset 
            +1 inputed into sklearn correspond to "fresh" in the dataset 
        
        To run call on the command line: 
            python3 chatbot.py --train_logreg_sentiment

        Hints: 
            - You do not need to write logistic regression from scratch (you did that in HW3). 
            Instead, look into the sklearn LogisticRegression class. We recommend using scratch.ipynb
            to get used to the syntax of sklearn.LogisticRegression on a small toy example. 
            - Review how we used CountVectorizer from sklearn in this code
                https://github.com/cs375williams/hw3-logistic-regression/blob/main/util.py#L193
            - You'll want to lowercase the texts
            - Our solution uses less than about 10 lines of code. Your solution might be a bit too complicated.
            - We achieve greater than accuracy 0.7 on the training dataset. 
        """ 
        #load training data  
        texts, y = util.load_rotten_tomatoes_dataset()

        self.model = None #variable name that will eventually be the sklearn Logistic Regression classifier you train 
        self.count_vectorizer = None #variable name will eventually be the CountVectorizer from sklearn 

        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                                
        data = {}

        # transform list y into list of -1 if 'rotten' and 1 if 'fresh'
        converted_y = [-1 if rating == 'Rotten' else 1 for rating in y]
        data['Y_train'] = np.array(converted_y)


        # convert texts
        texts = [text.lower() for text in texts]
        self.count_vectorizer = CountVectorizer(min_df=20, #only look at words that occur in at least 20 docs
                                stop_words='english', # remove english stop words
                                max_features=1000, #only select the top 1000 features 
                                ) 
        data['X_train'] = self.count_vectorizer.fit_transform(texts)

        print(data['X_train'].shape, data['Y_train'].shape)

        # training logreg on training data
        self.model = linear_model.LogisticRegression().fit(data['X_train'], data['Y_train']) 
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    def predict_sentiment_statistical(self, user_input: str) -> int: 
        """ 
        Uses a trained bag-of-words Logistic Regression classifier to classifier the sentiment

        In this function you'll also uses sklearn's CountVectorizer that has been 
        fit on the training data to get bag-of-words representation.

        Example 1:
            sentiment = chatbot.predict_sentiment_statistical('This is great!')
            print(sentiment) // prints 1 

        Example 2:
            sentiment = chatbot.predict_sentiment_statistical('This movie is the worst')
            print(sentiment) // prints -1

        Example 3:
            sentiment = chatbot.predict_sentiment_statistical('blah')
            print(sentiment) // prints 0

        Arguments: 
            - user_input (str) : a user-supplied line of text
        
        Returns: int 
            -1 if the trained classifier predicts -1 
            1 if the trained classifier predicts 1 
            0 if the input has no words in the vocabulary of CountVectorizer (a row of 0's)

        Hints: 
            - Be sure to lower-case the user input 
            - Don't forget about a case for the 0 class! 
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                             
        # lowercase
        user_input = user_input.lower()
        
        input = self.count_vectorizer.transform([user_input]).toarray()

        # predict output
        output = self.model.predict(input)
        if np.sum(input) == 0: return 0

        return output[0]
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    ############################################################################
    # 4. Movie Recommendation                                                  #
    ############################################################################

    def recommend_movies(self, user_ratings: dict, num_return: int = 3) -> List[str]:
        """
        This function takes user_ratings and returns a list of strings of the 
        recommended movie titles. 

        Be sure to call util.recommend() which has implemented collaborative 
        filtering for you. Collaborative filtering takes ratings from other users
        and makes a recommendation based on the small number of movies the current user has rated.  

        This function must have at least 5 ratings to make a recommendation. 

        Arguments: 
            - user_ratings (dict): 
                - keys are indices of movies 
                  (corresponding to rows in both data/movies.txt and data/ratings.txt) 
                - values are 1, 0, and -1 corresponding to positive, neutral, and 
                  negative sentiment respectively
            - num_return (optional, int): The number of movies to recommend

        Example: 
            bot_recommends = chatbot.recommend_movies({100: 1, 202: -1, 303: 1, 404:1, 505: 1})
            print(bot_recommends) // prints ['Trick or Treat (1986)', 'Dunston Checks In (1996)', 
            'Problem Child (1990)']

        Hints: 
            - You should be using self.ratings somewhere in this function 
            - It may be helpful to play around with util.recommend() in scratch.ipynb
            to make sure you know what this function is doing. 
        """ 
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################       

         # check if user_ratings has a count of at least 5. if yes, run util.recommend(). if not, raise error
        if len(user_ratings) < 5:
            raise ValueError("at least 5 ratings required to make a recommendation")
        
        # convert user_ratings into the same type and shape of self.ratings to use as argument for util.recommend
        num_movies = self.ratings.shape[0]
        user_ratings_array = np.zeros(num_movies)

        for movie_idx, rating in user_ratings.items():
            user_ratings_array[movie_idx] = rating

        # call util.recommend
        recommended_movie_indices = util.recommend(user_ratings_array, self.ratings, num_return)

        # take output of util.recommend(), and convert indices of movies into names of movies
        recommended_movies = [self.titles[i][0] for i in recommended_movie_indices]

        return recommended_movies
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    ############################################################################
    # 5. Open-ended                                                            #
    ############################################################################

    def spell_checker(self, user_input: str) -> str:
        """
        This function takes user_input and returns the spell-corrected string if the function 
        detects a spelling error. If there is no spelling error, the function returns the original string.

        This function must have a user_input argument to run. 

        

        Arguments: 
            - user_input (str): 
                - the str user_input that the user writes while interacting with Letterbot

        Example: 
            spell_corrected = chatbot.spell_checker('I liek "Avatar"')
            print(spell_corrected) // prints 'I like Avatar'

        Example: 
            spell_corrected = chatbot.spell_checker('I disliek "Kung Fu Panda 4" avd I lovi "The Martian"')
            print(spell_corrected) // prints 'I dislike "Kung Fu Panda 4" avd I love "The Martian"'
        """ 
    
        def load_wordlist(filepath: str) -> List[str]:
            with open(filepath, 'r') as f:
            # Read the file line by line and strip newline characters
                words = f.read().splitlines()
            return words

        # Load the 1000 most common words from 'data.txt'
        vocabulary = load_wordlist('deps/thousand_common.txt')
        # vocabulary = ['like', 'live', 'lives', 'look', 'liked', 'lie', 'life'] # potentially use a full vocab for spell checking
    
        # calculate hamming distance
        def hamming_distance(word1: str, word2: str) -> int:
            if len(word1) != len(word2):
                return float('inf')  # large num if not equal len
            
            # use zip to create char pairs at each index for words 
            match_chars = zip(word1, word2)

            # create list of True/False depending on if char is = 
            matching_or_not = [c1 != c2 for c1, c2 in match_chars]

            # return True's (1) + False's (0)
            return sum(matching_or_not)
        
        # use hamming_distance to correct the misspelled word
        def hamming_spell_check(word: str, vocabulary: List[str]) -> str:
            if any(c.isupper() for c in word) or word.startswith('"') or word.endswith('"'):
                return word  # Return the word as is
            
            # candidates are words that have same length as word...hamming
            candidates = [vw for vw in vocabulary if len(vw) == len(word)]
            if not candidates:
                return word  # return word as is

            # find w with smallest hamming distance to word and return
            return min(candidates, key=lambda w: hamming_distance(word, w))
        
        # apply spell checker
        corrected_words = [hamming_spell_check(token, vocabulary) for token in user_input.split()]
        return ' '.join(corrected_words)
    
    def clean_articles(self, titles: list): 
        """
        function2.  
        """
        def lowercase_articles(title: str) -> str:
            articles = ['the', 'a', 'an']
            words = title.split()
            
            # Skip the first word, which we want to keep uppercase
            for i in range(1, len(words)):
                if words[i].lower() in articles:
                    words[i] = words[i].lower()
            
            return ' '.join(words)

        for index, title in enumerate(titles):
            title = lowercase_articles(title)
            titles[index] = title.replace("The ", "").replace("An " ,"").replace("A ", "")

        return titles

    def function3(self, user_input: str) -> str:
        """
        function3. 
        """
        # call predict_sentiment_rule_based on the user's message
        predicted_sentiment = self.predict_sentiment_rule_based(user_input)
        
        # same regex use from predict_sentiment_rule_based
        regex = r'\b\w+\b'
        tokens = re.findall(regex, user_input.lower())
        
        # save all tokens
        sentiment_tokens = []
        anti_sentiment_tokens = []

        # convert numerical sentiment val into str val so comparison is easier
        if predicted_sentiment == 1:
            predicted_sentiment = 'pos'
        elif predicted_sentiment == -1:
            predicted_sentiment = 'neg'
        else: 
            predicted_sentiment = 'reg'

        # add token to sentiment or anti_sentiment list
        for token in tokens:
            if token in self.sentiment:
                if self.sentiment[token] == predicted_sentiment:
                    sentiment_tokens.append(token)
                else:
                    anti_sentiment_tokens.append(token)

        sentiment_str = ""
        if len(sentiment_tokens) == 1:
            sentiment_str = sentiment_tokens[0]
        elif len(sentiment_tokens) == 2:
            sentiment_str = f"{sentiment_tokens[0]} and {sentiment_tokens[1]}"
        elif len(sentiment_tokens) > 2:
            sentiment_str = ", ".join(sentiment_tokens[:-1]) # join all but last token with ,
            sentiment_str += f", and {sentiment_tokens[-1]}" # add last sentiment with , and for proper grammar

        anti_sentiment_str = ""
        if len(anti_sentiment_tokens) == 1:
            anti_sentiment_str = anti_sentiment_tokens[0]
        elif len(anti_sentiment_tokens) == 2:
            anti_sentiment_str = f"{anti_sentiment_tokens[0]} and {anti_sentiment_tokens[1]}"
        elif len(anti_sentiment_tokens) > 2:
            anti_sentiment_str = ", ".join(anti_sentiment_tokens[:-1]) # join all but last token with ,
            anti_sentiment_str += f", and {anti_sentiment_tokens[-1]}" # add last sentiment with , and for proper grammar

        result = ''
        if predicted_sentiment == 'neg':
            if not anti_sentiment_tokens: 
                result = f"I'm sorry to hear that you are {sentiment_str}. Here is a funny comedy movie that might cheer you up: "". I hope the rest of your day is better!"
            else:
                result = f"I'm sorry to hear that you are {sentiment_str}, but I am glad to hear you are at least {anti_sentiment_str}. Here is a funny comedy movie that might cheer you up: "" . I hope the rest of your day is better!"

        elif predicted_sentiment == 'pos':
            if not anti_sentiment_tokens: 
                result = f"I am glad you are feeling {sentiment_str}! It must be the autumn leaves, the foliage is beautiful."
            else:
                result = f"I am glad you are feeling {sentiment_str}! It must be the autumn leaves, the foliage is beautiful. But I'm sorry to hear you are {anti_sentiment_str}, hopefully you day gets better."
        else:
            result = f"Thanks for letting me know."
        
        return result 

if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')



