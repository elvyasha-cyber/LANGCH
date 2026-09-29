"""
Агент возвращает список фактов о породе кошек или собак.

Снаружи виден только список. У каждого пользователя свой thread_id:
его история не смешивается с чужой. Это скрыто внутри ask().
"""

import os
import re
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

load_dotenv()

SYSTEM_PROMPT = """Ты серьёзный справочник по породам кошек и собак.
Отвечай только фактами, без шуток, приветствий и вводных фраз.
Если вопрос о кошках, вызови get_cat_fact.
Если вопрос о собаках, вызови get_dog_fact.
Перенеси полученные факты в список и ничего больше."""

FACTS = {
    "золотая шиншилла": [
        "Золотая шиншилла — это окрас шиншиллового типа с тёплым золотистым подтоном, а не всегда отдельная порода.",
        "Кончики остевых волос окрашены, а нижняя часть светлая, поэтому шерсть выглядит мерцающей.",
        "У кошек этого окраса обычно зелёные глаза.",
        "Шерсть плотная и нуждается в регулярном вычёсывании.",
    ],
    "сиамская": [
        "Сиамская кошка — порода с колор-пойнтом: корпус светлый, а уши, маска, лапы и хвост темнее.",
        "Глаза ярко-голубые.",
        "Тело мускулистое и вытянутое, голова клиновидная.",
        "Характер активный, кошки голосистые и сильно привязываются к человеку.",
    ],
}


@tool
def get_cat_fact(breed: str) -> str:
    """Возвращает один или несколько фактов о породе кошек. Аргумент breed — название породы."""
    key = breed.strip().lower().replace("ё", "е")
    for name, facts in FACTS.items():
        if name in key or key in name:
            return "\n".join(facts)
    return _facts_from_model("кошек", breed)


def _facts_from_model(animal: str, breed: str) -> str:
    reply = model.invoke(
        f"Дай 4 коротких серьёзных факта о породе {animal} "
        f"«{breed}». По одному факту на строку. "
        "Без нумерации, шуток и вступлений."
    )
    content = reply.content
    if isinstance(content, str):
        return content.strip()
    lines = []
    for block in content:
        if isinstance(block, str):
            lines.append(block)
        elif isinstance(block, dict) and block.get("text"):
            lines.append(str(block["text"]))
    return "\n".join(lines).strip()


@tool
def get_dog_fact(breed: str) -> str:
    """Возвращает один или несколько фактов о породе собак. Аргумент breed — название породы."""
    return _facts_from_model("собак", breed)


model = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0,
    timeout=300,
    max_tokens=1500,
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)


@dataclass
class Context:
    """Кто задаёт вопрос. В историю диалога это не пишется, передаётся на вызов."""

    user_id: str


@dataclass
class ResponseFormat:
    """Строгий ответ агента: только список фактов."""

    facts: List[str]


checkpointer = InMemorySaver(
    serde=JsonPlusSerializer(
        allowed_msgpack_modules=[("__main__", "ResponseFormat")],
    )
)

agent = create_agent(
    model=model,
    tools=[get_cat_fact, get_dog_fact],
    system_prompt=SYSTEM_PROMPT,
    response_format=ToolStrategy(ResponseFormat),
    context_schema=Context,
    checkpointer=checkpointer,
)


def ask(text: str, user_id: str) -> tuple[List[str], str]:
    """Свой диалог на каждого user_id. Список фактов и мордочка: кошка или собака."""
    response = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": user_id}},
        context=Context(user_id=user_id),
    )
    mark = "🐱"
    for message in response["messages"]:
        if getattr(message, "name", None) == "get_dog_fact":
            mark = "🐶"
        elif getattr(message, "name", None) == "get_cat_fact":
            mark = "🐱"
    return response["structured_response"].facts, mark


def show(facts: List[str], mark: str) -> None:
    """Каждое предложение — отдельная строка со своей мордочкой."""
    for fact in facts:
        parts = re.split(r"(?<=[.!?])\s+", fact.strip())
        for sentence in parts:
            sentence = sentence.strip()
            if sentence:
                print(f"{mark} {sentence}")


if __name__ == "__main__":
    user_id = input("Номер пользователя: ").strip() or "1"
    print("Можно спрашивать о кошках и о собаках. Пустая строка или «выход» завершают диалог.")
    while True:
        try:
            text = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() in {"выход", "exit", "quit"}:
            break
        facts, mark = ask(text, user_id)
        show(facts, mark)
        print()
