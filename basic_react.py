from openai import OpenAI
import re
import os

from dotenv import load_dotenv
_ = load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


class Agent:
    def __init__(self, system=''):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({
                'role': 'system', 'content': system,
            })
    def __call__(self, message):
        self.messages.append({'role': 'user', 'content': message})
        result = self.execute()
        self.messages.append({'role': 'assistant', 'content': result })
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model='gpt-4o',
            temperature=0,
            messages=self.messages,
        )
        return completion.choices[0].message.content

prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

get_book_genre:
e.g. get_book_genre: The Catcher in the Rye
Returns the genre(s) of the given book.

get_similar_books:
e.g. get_similar_books: 1984
Returns a list of 3–5 books similar to the one given.

get_books_by_author:
e.g. get_books_by_author: Haruki Murakami
Returns a list of notable books written by that author.

Example session:

Question: I loved reading 1984 by George Orwell. What should I read next?
Thought: I should get similar books to 1984.
Action: get_similar_books: 1984
PAUSE

(Then, after being called again with...)

Observation: Similar books to 1984 include Brave New World, Fahrenheit 451, and We.

Answer: If you enjoyed *1984*, you might love *Brave New World* by Aldous Huxley or *Fahrenheit 451* by Ray Bradbury. They explore similar dystopian themes.
""".strip()

def get_book_genre(book: str) -> str:
    _system_prompt = "Return the literary genre(s) of the book mentioned."
    _user_prompt = f"The book:\n{book}"
    chat_response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "system", "content": _system_prompt}, {"role": "user", "content": _user_prompt}],
        temperature=0,
        max_tokens=40,
    )
    return chat_response.choices[0].message.content.strip()

def get_similar_books(book: str) -> str:
    _system_prompt = "Suggest 3–5 books that are similar in theme, style, or genre to the given book."
    _user_prompt = f"The book:\n{book}"
    chat_response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "system", "content": _system_prompt}, {"role": "user", "content": _user_prompt}],
        temperature=0,
        max_tokens=100,
    )
    return chat_response.choices[0].message.content.strip()

def get_books_by_author(author: str) -> str:
    _system_prompt = "List 3–5 notable books written by the given author."
    _user_prompt = f"The author:\n{author}"
    chat_response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "system", "content": _system_prompt}, {"role": "user", "content": _user_prompt}],
        temperature=0,
        max_tokens=100,
    )
    return chat_response.choices[0].message.content.strip()

known_actions = {
    'get_book_genre': get_book_genre,
    'get_similar_books': get_similar_books,
    'get_books_by_author': get_books_by_author,
}


action_re = re.compile('^Action: (\w+): (.*)$')

def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question

    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]

        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception(f'Unknown action: {action} : {action_input}')

            print('-- running {} {}'.format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)

            next_prompt = 'Observation: {}'.format(observation)

        else:
            return

# --------- User question -------------------------------------
question = "I loved watching the Sandman on Netflix. It is based on books. What genre is it, and can you suggest similar books? Also, what other books did it's author write?"
query(question)