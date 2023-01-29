# import necessary modules
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

from PIL import Image
import warnings

from slow_model.model import run_model
from fast_model.fast_model_tf import fast_model
from config import TOKEN

warnings.filterwarnings('ignore')

bot = Bot(token=TOKEN)  # Set bot with our telegram-bot token
dp = Dispatcher(bot)  # Set dispatcher

# Create buttons for our bot
button_slow_transform = KeyboardButton('Slow transformation\U0001F422 (but more quality)')
button_fast_transform = KeyboardButton('Fast transformation\U0001F3C3 (but less quality, maybe...)')
button_text_slow = 'Slow transformation\U0001F422 (but more quality)'
button_text_fast = 'Fast transformation\U0001F3C3 (but less quality, maybe...)'

# Initialize keyboard with our buttons
kb = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
kb.add(button_slow_transform)
kb.add(button_fast_transform)


# Add message handlers for react to input messages
@dp.message_handler(commands=['start'])
async def welcome(message: types.Message):
    """
    Answer on /start message

    Parameters
    ----------
    Message:
        Input text message
    """

    await message.answer(f"Hello \U0001F44B, I'm Transform bot! \nAre you ready for transform your picture? \
    \nPlease choose type of transformation \U0001F914", reply_markup=kb)


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    """
    Answer on /help message

    Parameters
    ----------
    Message:
        Input text message
    """

    await message.answer("\U00002139 Send me two pictures \U0001F5BC: content and style. \
    \nAfter that you'll get picture after style transfer operation\U0001F609\
    \nSend me /start for starting style transfer process\U0001F680")


@dp.message_handler()
async def transform(message: types.Message):
    """
    Answer on text message from buttons

    Parameters
    ----------
    Message:
        Input text message
    """

    global button_text
    button_text = message.text
    if button_text in [button_text_slow, button_text_fast]:
        await message.answer(
            "Please send me two images \U0001F5BC \U0001F305: \n\U00002705 Content image (what you want to transform)\
            \n\U00002705 Style image (what style you want to get)")
        await message.answer("Now load content image\U0001F5BC")
    return button_text


img_list = []


@dp.message_handler(content_types=['photo'])
async def get_photo(message):
    """
    Get the photo and start style transfer process

    Parameters
    ----------
    Message:
        Input photo
    """

    file_info = await bot.get_file(message.photo[-1].file_id)
    new_photo = await bot.download_file(file_info.file_path)  # get input image
    image = Image.open(new_photo)  # transform to PIL format
    img_list.append(image)  # save input image to list
    if len(img_list) % 2 != 0:
        await message.answer("Now load style image\U0001F305")
    if len(img_list) % 2 == 0:
        if button_text == button_text_slow:  # go to the slow variant of style transfer model
            await message.answer("it will be ready in about several minutes...\U000023F3")
            image = run_model(content_image=img_list[-2], style_image=img_list[-1])
            await bot.send_photo(message.from_user.id, image,
                                 caption='Your awesome transformed image \U0001F307')
        elif button_text == button_text_fast:  # got to the fast variant of style transfer model
            await message.answer("it will be ready in about several seconds...\U000023F1")
            image = fast_model(content_image=img_list[-2], style_image=img_list[-1])
            await bot.send_photo(message.from_user.id, image,
                                 caption='Your awesome transformed image \U0001F307')
        else:
            await message.answer("Please send me /start or /help!")


if __name__ == '__main__':
    executor.start_polling(dp)
