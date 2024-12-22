import aiohttp
import asyncio
import logging
import os
import openai
import openpyxl
import re
import textwrap
import time
from datetime import datetime

from aiogram import Bot, Dispatcher, Router, types, F
from aiogram.filters import Command
from aiogram.types import (
    InlineKeyboardMarkup, InlineKeyboardButton as IKB,
    KeyboardButton as KB, ReplyKeyboardMarkup,
    Message, CallbackQuery, BotCommand
)
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.filters.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
#from aiogram.filters import ContentTypeFilter # Import ContentTypeFilter if needed
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv



import requests
import logging

logger = logging.getLogger(__name__)


@router.callback_query(F.data.in_(SECOND_LEVEL_BUTTONS))
async def handle_second_level_buttons(call: CallbackQuery):
    logger.info(f"Button clicked: {call.data} by user {call.from_user.id}")
    # Остальной код...

load_dotenv()  # Load environment variables from .env file
drive.mount('/content/drive')

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - [%(levelname)s] - %(name)s - "
                           "(%(filename)s).%(funcName)s(%(lineno)d) - %(message)s")

import os
os.environ["OPENAI_API_KEY"] = "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"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Изменено здесь

# Проверяем, был ли найден ключ
if OPENAI_API_KEY is None:
  raise ValueError("Переменная окружения OPENAI_API_KEY не установлена.")


# Constants
TOKEN = '7490574376:AAE_r9y3KNsz-xc57K1NXnSjIBYyFVO9Dgk'
user_data = {}  # Dictionary to store user data
# Global variables
media_files = {}
document_files = {}
dialog_history = {}
faiss_db = None
current_model = "gpt-3.5-turbo"
temp = 1
verbose = 0
relevant_chanks = 8

def similarity_search(query, search_index, k=1):
    return search_index.similarity_search(query=query, k=k)


def answer_neuro(system_neuro, instruction_neuro, topic, search_index, summary_history, temp=1, verbose=0, k=8, model="gpt-3.5-turbo"):
    docs = search_index.similarity_search(query=topic, k=k)
    message_content = re.sub(r'\r\n', ' ', '\n '.join([f'\n--------------------\n' + doc.page_content + '\n' for doc in docs]))
    # Define faiss_response here using the retrieved documents (docs)
    faiss_response = '\n'.join([doc.page_content for doc in docs])  # Define faiss_response

    if verbose:
        print('Чанки :\n ======================================== \n', message_content)

    messages = [
        {"role": "system", "content": system_neuro},
        {"role": "user", "content": f"{instruction_neuro}.\nКонтекст из базы знаний FAISS:\n{faiss_response}.\n\nВопрос:\n{topic}\n\nХронология предыдущих сообщений диалога: {summary_history}\n\nОтвет:"}
    ]

    start_time = time.time()
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temp
    )
    end_time = time.time()

    answer = completion.choices[0].message.content
    print('\n==================')
    print("Агент консультант:")
    formatted_answer = textwrap.fill(answer, width=120)
    print(formatted_answer)

    price = round((5 * completion['usage']['prompt_tokens'] / 1000000) + (15 * completion['usage']['completion_tokens'] / 1000000), 5)
    print('ЦЕНА запроса:', price, '$') # Fixed: Removed extra ')' and added '$' for consistency
    response_time = end_time - start_time
    print(f"Время ответа: {end_time - start_time:.2f} секунд")

    return answer


embeddings = OpenAIEmbeddings()  # Initialize embeddings here
faiss_db = FAISS.load_local("faiss_index", embeddings) # Removed allow_dangerous_deserialization=True

# Загрузка индекса FAISS
async def load_faiss():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local("faiss_index", embeddings) # Removed allow_dangerous_deserialization=True

faiss_db = None

faiss_db = None
# --- Bot Setup ---
bot = Bot(token=TOKEN)
dp = Dispatcher()
router = Router()
dp.include_router(router)


# --- Command Handlers ---
@router.message(Command("start"))
async def cmd_start(message: types.Message):
    """Handles the /start command."""
    await message.answer("Я - нейро-сотрудник, созданный для помощи в "
                         "организации системы корпоративного здоровья "
                         "на Вашем предприятии")


@router.message(Command("dialog"))
async def handle_dialog_command(message: Message):
    """Handles the /dialog command."""
    await message.answer(
        "Вы можете задать вопрос, выбрать специализацию, и я предоставлю "
        "Вам доступную информацию.",
        reply_markup=create_control_buttons()
    )


@router.message(Command("document")) # /document to see the list of documents
async def send_document_list(message: types.Message):
    """Sends a list of available documents to the user."""
    if not document_files:
        await message.answer("Нет доступных документов.")
        return

    document_list = "\n".join([f"/{key}: {value['description']}" for key, value in document_files.items()])
    await message.answer(f"Доступные документы:\n{document_list}")



@router.message(Command("help"))
async def cmd_help(message: types.Message):
    """Handles the /help command."""
    await message.answer(
        "Вот список доступных команд:\n"
        "/start - начать работу со мной\n"
        "/dialog - нейро-сотрудник ответит на Ваши вопросы\n"
        "/help - возможности бота\n"
        "/pay - оплата\n"
        "/about - информация обо мне."
    )

async def set_commands(bot: Bot):
    """Sets bot commands in Telegram interface."""
    commands = [
        BotCommand(command="/start", description="Начать диалог"),
        BotCommand(command="/dialog", description="Задать вопрос"),
        BotCommand(command="/help", description="Помощь"),
        BotCommand(command="/about", description="О боте"),
    ]
    await bot.set_my_commands(commands)


# Обработчик команды /about
@router.message(Command("about"))
async def cmd_about(message: types.Message):
    await message.answer(
        "Нейро-сотрудник CorpHealth - диалоговая система ИИ для организации системы корпоративного здоровья работников.\n"
        "Правообладатель: команда разработчиков ООО Портал РАМН.\n"
        "Адрес главного офиса ООО Портал РАМН: г. Москва, ул. Верхняя Красносельская, д. 20/1.\n"
        "Телефон: +7 499 455 0603 Время работы: Пн-Пт: 09.00-18.00\n"
        "E-mail: info@portalramn.ru"
    )

# --- Inline Keyboard Handling ---
def create_control_buttons():
    """Creates the main inline keyboard."""
    buttons = [
        [IKB(text="Нейро-ПрМО", callback_data="neuro_prmo")],
        [IKB(text="Нейро-медэксперт", callback_data='neuro_med_expert')],
        [IKB(text="Нейро-юрист", callback_data='neuro_lawyer')],
        [IKB(text="Нейро-HR", callback_data='neuro_hr')],
        [IKB(text="Нейро-сотрудник ОСКЗ", callback_data='neuro_okz')],
        [IKB(text="Нейро-IT", callback_data='neuro_IT')],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def create_buttons_level_two(category):
    """Creates the second-level inline keyboard based on category."""
    buttons_dict = {
        'neuro_prmo': [
            [IKB(text="Чек-лист ПрМО", callback_data='чек-лист ПрМО')],
            [IKB(text="Выбор медорганизации", callback_data='выбор медорганизации')],
            [IKB(text="Выбор терминала ПАК", callback_data='выбор терминала ПАК')],
        ],
        'neuro_med_expert': [
            [IKB(text="Выбор медорганизации для ОСКЗ", callback_data='выбор медорганизации для ОСКЗ')],
            [IKB(text="Договор с медорганизацией", callback_data='договор с медорганизацией')],
            [IKB(text="Учет медосмотров", callback_data='учет медосмотров')],
            [IKB(text="Управление рисками здоровью на предприятии", callback_data='управление рисками')],
        ],
        'neuro_lawyer': [
            [IKB(text="Нормативно-правовые документы", callback_data='нормативно-правовые документы')],
            [IKB(text="Внутренняя документация на предприятии", callback_data='внутренняя документация')],
            [IKB(text="Взаимодействие с регуляторами", callback_data='взаимодействие с регуляторами')],
        ],
        'neuro_hr': [
            [IKB(text="Протокол организации системы КЗ", callback_data='протокол ОСКЗ')],
            [IKB(text="Должностные инструкции", callback_data='должностные инструкции')],
            [IKB(text="Обучение сотрудников", callback_data='обучение сотрудников')],
        ],
        'neuro_okz': [
            [IKB(text="Инструкция ОСКЗ", callback_data='инструкция ОСКЗ')],
            [IKB(text="Документация и отчётность по ОКЗ", callback_data='документация и отчётность по ОКЗ')],
            [IKB(text="Бизнес-процессы ОСКЗ", callback_data='бизнес-процессы ОСКЗ')],
            [IKB(text="Мероприятия СКЗ", callback_data='мероприятия СКЗ')],
        ],
        'neuro_IT': [
            [IKB(text="Задачи Нейро-IT-сотрудника", callback_data='задачи Нейро-IT')],
            [IKB(text="Функциональные возможности ИС КЗ", callback_data='функциональные возможности ИС КЗ')],
            [IKB(text="Интеграции ИС КЗ", callback_data='интеграции ИС КЗ')],
            [IKB(text="Техническое задание ИС КЗ", callback_data='техническое задание ИС КЗ')],
        ]
    }
    return InlineKeyboardMarkup(inline_keyboard=buttons_dict.get(category, []))


# Callback data constants
FIRST_LEVEL_BUTTONS = ['neuro_prmo', 'neuro_med_expert', 'neuro_lawyer', 'neuro_hr',
                       'neuro_okz', 'neuro_IT']
SECOND_LEVEL_BUTTONS = [
    'чек-лист ПрМО', 'выбор медорганизации', 'выбор терминала ПАК',
    'выбор медорганизации для ОСКЗ', 'договор с медорганизацией',
    'учет медосмотров', 'управление рисками',
    'нормативно-правовые документы', 'внутренняя документация',
    'взаимодействие с регуляторами',
    'протокол ОСКЗ', 'должностные инструкции',
    'обучение сотрудников',
    'инструкция организации СКЗ', 'документация и отчётность по ОСКЗ',
    'бизнес-процессы ОСКЗ', 'мероприятия СКЗ',
    'задачи Нейро-IT', 'функциональные возможности ИС КЗ',
    'интеграции ИС КЗ', 'техническое задание ИС КЗ',
]


@router.callback_query(F.data.in_(FIRST_LEVEL_BUTTONS))
async def handle_first_level_buttons(call: CallbackQuery):
    """Handles clicks on the first-level inline keyboard buttons."""
    response_text = {
        'neuro_prmo': "Вы выбрали дистанционный предрейсовый медосмотр. "
                "Уточните Ваш выбор нажатием кнопки либо задайте свой вопрос.",
        'neuro_med_expert': "Вы выбрали нейро-медэксперта. "
                            "Уточните Ваш выбор нажатием кнопки либо "
                            "задайте свой вопрос.",
        'neuro_lawyer': "Вы выбрали нейро-юриста. "
                        "Уточните Ваш выбор нажатием кнопки либо "
                        "задайте свой вопрос.",
        'neuro_hr': "Вы выбрали нейро-HR. "
                    "Уточните Ваш выбор нажатием кнопки либо "
                    "задайте свой вопрос.",
        'neuro_okz': "Вы выбрали нейро-ОКЗ. "
                     "Уточните Ваш выбор нажатием кнопки либо "
                     "задайте свой вопрос.",
        'neuro_IT': "Вы выбрали нейро-IT. "
                    "Уточните Ваш выбор нажатием кнопки либо "
                    "задайте свой вопрос.",
    }.get(call.data)
    if response_text:
        await call.message.answer(response_text,
                                 reply_markup=create_buttons_level_two(call.data))
    await call.answer()


from aiogram.types import InputFile  # Убедитесь, что вы импортируете InputFile

@router.callback_query(F.data.in_(SECOND_LEVEL_BUTTONS))
async def handle_second_level_buttons(call: CallbackQuery):
    """Handles clicks on the second-level inline keyboard buttons."""
    await call.message.answer(f"Вы выбрали: {call.data}. Уточните Ваш вопрос.")
    await call.answer()

# --- Main Message Handler ---


@router.message(F.text & ~(F.text == "/start") &
                ~(F.text == "/dialog") & ~(F.text == "/help") &
                ~(F.text == "/pay") & ~(F.text == "/about"))
async def handle_text_message(message: types.Message, topic: str = None):
    global summary_history, faiss_db, current_model
    logging.info(f"handle_text_message called with message: {message}, topic: {topic}")

    user_id = message.from_user.id
    logging.info(f"User ID: {user_id}")

    # Проверка существования данных пользователя
    if user_id not in user_data:
        user_data[user_id] = {"name": None}

    # Извлечение имени, если оно указано
    name_match = re.search(r"меня зовут (\w+)", message.text, re.IGNORECASE)
    if name_match:
        user_data[user_id]["name"] = name_match.group(1)
        await message.answer(f"Приятно познакомиться, {user_data[user_id]['name']}! "
                             "Готов помочь обсудить организацию системы "
                             "корпоративного здоровья в вашей компании.")
        return  # Выход из функции после установки имени

    if user_id not in dialog_history:
        dialog_history[user_id] = []
    dialog_history[user_id].append(message.text)

    summary_history = "\n".join(dialog_history[user_id])

    context = "context is not available"  # Значение по умолчанию

    # Получаем релевантные документы
    docs = faiss_db.similarity_search(query=message.text, k=3)
    if docs:
        context = "\n".join([doc.page_content for doc in docs])  # Извлечение контекста

    name = user_data[user_id]["name"] or "Вы"
    prompt = (f"{system_neuro}{instruction_neuro}\n"
                f"Контекст:\n{context}\nВопрос:\n{message.text}\nИмя: {name}")

    processing_message = await message.answer("Обработка сообщения...")

    try:
        start_time = time.time()

        # 1. Запрос в базу данных FAISS
        docs = faiss_db.similarity_search(message.text, k=relevant_chanks)

        # 2. Извлечение контекста
        context = "\n".join([doc.page_content for doc in docs])

        # 3. Подготовка запроса для LLM
        prompt = (f"{system_neuro}{instruction_neuro}\n"
                  f"Контекст:\n{context}\nВопрос:\n{message.text}")

        # 4. Вызов LLM (OpenAI)
        response = openai.ChatCompletion.create(
            model=current_model,
            messages=[
                {"role": "system", "content": system_neuro},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=500,
        )
        llm_answer = response.choices[0].message.content.strip()

        # Расчет времени ответа и цены
        end_time = time.time()
        response_time = end_time - start_time
        price = round((5 * response['usage']['prompt_tokens'] / 1000000) +
                      (15 * response['usage']['completion_tokens'] / 1000000), 7)
        formatted_price = "{:.7f}".format(price)

        # Отправка ответа пользователю
        await bot.edit_message_text(
            text=f"{llm_answer}\n\nЦена за этот запрос: {formatted_price}",
            chat_id=processing_message.chat.id,
            message_id=processing_message.message_id,
        )
                # Simulate a message object for consistency


        # Log the interaction to Excel
        ws.append([
            message.from_user.id,
            message.from_user.username,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            message.text,
            llm_answer,
            response_time,
            price
        ])
        await asyncio.to_thread(wb.save, 'dialog_history4.xlsx')  # Save Excel file in a separate thread

    except Exception as e:
        logging.error(f"Error processing message: {e}")
        await bot.edit_message_text(
            text=f"Произошла ошибка: {e}",
            chat_id=processing_message.chat.id,
            message_id=processing_message.message_id,
        )

# --- System Prompts ---
system_neuro = """Вы - нейро-сотрудник профессиональный эксперт в области корпоративного здоровья.
Ваша цель как нейро-сотрудника компании помочь организовать систему управления
бизнес-процессами укрепления и развития корпоративного здоровья каждому
специалисту компании, в лице пользователей.
Действуйте всегда от имени конкретного агента в рамках агентской схемы,
прописанной в Промте нейро-сотрудника.
Запомни: ОСКЗ - организация системы корпоративного здороья.
Это агенты. которые отвечают за конкретные задачи на предприятии:
"Нейро-ПрМО" - это агент-специалист по организации дистанционного предрейсового/послерейсового осмотра водителей.
"Нейро-медэксперт" - это агент-специалист обеспечивающий связь между медорганизацией и предприятием на предмет организации системы корпоративного здоровья.
"Нейро-юрист" - это агент-специалист по обеспечению правовового сопровождения ОСКЗ на предприятии.
"Нейро-HR" - это агент-специалист, отвечающий за кадровое обеспечение организации системы корпоратвного здоровья (ОСКЗ) на предприятии.
"Нейро-сотрудник ОКЗ" - это агент-специалист ответсвенный за организацию системы корпоративного здороья на предприятии (ОСКЗ).
"Нейро-IT" - это агент-специалист, отвечающий за организацию модуля информационной системы корпоративного здоровья на предприятии.
Стилистика общения: Общайся простым языком, без формальностей.
Обращайся к клиенту всегда на Вы.
Запрещено здороваться с пользователем, запрещено называть пользователя
Пользователем.
Запрещено спрашивать - "Как вас называть?"
Если клиент оставил контакты и сообщил как хочет реализовать проект, то
запрещено повторно запрашивать эту информацию.
Запрещено общаться на стороннюю тему. Если Пользователь задает стороннюю тему,
спрашивает не по теме организации системы корпоративного здоровья на
предприятии, не по материалам и продуктам Компании, ты категорически
отказываешься отвечать.
Запрещено упоминать в ответе, что ты анализировал отрывки документов и брал
оттуда информацию."""

instruction_neuro = """Проанализируй весь предыдущий диалог чтобы написать свой ответ
последовательным и логичным.
Категорически запрещено повторяться и здороваться.
Используй информацию только из приложенного документа базы FAISS,
Нельзя упоминать об Агентах общего назначения в ответе пользователю.
Нельзя называть Пользователя - Пользователем. Спроси как его называть, если
он не представился обращайся безлично.
Стилистика ответа должна быть поддерживающей беседу в контексте важности и
полезности формирования системы корпоративного здоровья
Если вы не знаете ответа, просто скажите что не знаете не пытайтесь
придумывать ответ
Ищи в папках и в интернте фотографии по запросу
В ответе нужно написать только ответ пользователю
Запрещено спрашивать пользователя что его еще интересует."""

# --- Excel Setup ---
try:
    wb = openpyxl.load_workbook('dialog_history4.xlsx')
except FileNotFoundError:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["UserID", "Username", "Timestamp", "Question", "Answer",
              "Response Time (s)", "Price ($)"])
    wb.save('dialog_history4.xlsx')
ws = wb.active

faiss_db = FAISS.load_local("faiss_index", embeddings) # Removed allow_dangerous_deserialization=True

# Функция остановки всех активных сессий
async def stop_previous_sessions():
    global active_tasks
    for task in active_tasks:
        if not task.done():
            task.cancel()  # Отменяем задачу
    active_tasks.clear()  # Очищаем список активных задач

# --- Main Function ---
# Глобальный список активных задач
active_tasks = []

async def main():
    """Main function to initialize and start the bot."""
    global faiss_db, embeddings

    await set_commands(bot)
    embeddings = OpenAIEmbeddings()
    faiss_db = await load_faiss()
    await bot.delete_webhook(drop_pending_updates=True)

    try:
        bot_info = await bot.get_me()
        logging.info(f"Bot info: {bot_info.username}")
    except Exception as e:
        logging.error(f"Error during bot initialization: {e}")
        return

    logging.info("Starting polling...")

    try:
        await dp.start_polling(bot)  # Start polling to listen for updates
    except Exception as e:
        logging.error(f"Error during polling: {e}")


# Modified to use asyncio.get_event_loop()
if __name__ == '__main__':
    import nest_asyncio
    nest_asyncio.apply()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
