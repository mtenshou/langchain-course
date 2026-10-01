from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv, find_dotenv
from pathlib import Path

import os

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# path = find_dotenv()
# print("found .env at:", repr(path))        # '' means not found
# print("loaded:", load_dotenv(path, override=True))
# print("key set:", bool(os.getenv("OPENAI_API_KEY")))



def main():
    print("Hello from langchain-course!")
    information = """Sadie Elizabeth Sink (born April 16, 2002) is an American actress. She began her career in theater as a child, playing the title role in the musical Annie (2012–2014) and young Elizabeth II in the historical play The Audience (2015) on Broadway. In 2016, she made her film debut in the biographical sports drama Chuck. Sink had her breakthrough portraying Max Mayfield in the Netflix science fiction horror series Stranger Things (2017–2025), for which she received critical praise.

    Sink appeared in the horror film trilogy Fear Street in 2021 and starred in Darren Aronofsky's 2022 psychological drama The Whale. She portrays Jean Grey in the Marvel Cinematic Universe (MCU), starting with the film Spider-Man: Brand New Day (2026), and will reprise the role in Avengers: Secret Wars (2027) and the untitled X-Men film (2028). She received a Tony Award nomination for Best Actress in a Play for her performance in the Broadway production of John Proctor Is the Villain (2025), and made her West End debut in Robert Icke's production of Romeo and Juliet in 2026.

    Early life
    Sadie Elizabeth Sink[1] was born on April 16, 2002,[2] in Brenham, Texas.[1][3] She is one of five children of Lori and Casey Sink and has three older brothers—Caleb, Spencer, and Mitchell—and a younger sister, Jacey.[4] Although her family was primarily interested in sports, Sink and Mitchell developed an interest in musical theater. They staged performances at home and watched Broadway productions and Tony Award performances together.[3][5]

    Sink made her stage debut at age seven as an ensemble member in a community-theater production of The Best Christmas Pageant Ever in Brenham.[3] The following year, she played Mary Lennox in an A.D. Players production of The Secret Garden. She later credited the experience with inspiring her to pursue acting professionally.[3][6] She trained at the Humphreys School of Musical Theatre at Theatre Under the Stars and attended a summer program at the Houston Family Arts Center.[6] In 2012, after Sink and Mitchell began obtaining professional stage work, their family relocated to New York City to support their careers.[6][4]

    As her acting career developed, Sink alternated between homeschooling, remote study, and public school. She attended public school during eighth grade and part of her first two years of high school before completing her final two years remotely.[7]
    """

    summary_template = """
    given the information {information} about a person I want you to create
    1. A short summary
    2. two interesting facts about him/her
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5.4-mini")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)
   

if __name__ == "__main__":
    main()
