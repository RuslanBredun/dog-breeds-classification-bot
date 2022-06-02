import telebot
import os
import cv2
import YOLO_detector as yolo
import config as cfg
import numpy as np


bot = telebot.TeleBot(cfg.token)
detector = yolo.Detector(cfg)


@bot.message_handler(commands=['helpme', 'start', 'info', '/help'])
def start(message):

    if message.text in ['/helpme', '/start', '/help']:
        bot.send_message(message.chat.id, f'Hello, {message.from_user.first_name}. You can type:\n'
                                          f'<b>/help, /helpme, /start</b> to check help; \n'
                                          f'<b>/info</b> to check what objects i can recognize; \n '
                                          f'<b>Drop the photo</b> and I can try to recognize something.',
                         parse_mode='html')
    elif message.text == '/info':
        bot.send_message(message.chat.id, f'So, {message.from_user.first_name}. I can recognize next object: \n'
                                          f'{detector.clNames}')
        bot.send_message(message.chat.id, f'Also I can recognize your dogs breed with ~85% accuracy\n'
                                          f'{detector.dog_breeds}')


@bot.message_handler()
def get_user_text(message):
    if message.text == 'Hello':
        bot.send_message(message.chat.id, f'Hello, {message.from_user.first_name}, how are you? ')
    elif message.text == 'Photo':
        bot.send_message(message.chat.id, f'No no no, {message.from_user.first_name}! You should send me a photo')


@bot.message_handler(content_types=['photo'])
def get_user_content(message):

    source_img, path = get_photo_from_msg(message)
    img, objs, bboxes = detector.find_objects(source_img)
    processed_path = save_photo(img, path)

    if objs:
        bot.send_photo(message.chat.id, open(processed_path, 'rb'))
        msgs = [str(unique + ' - ' + str(count)) for unique, count in zip(*np.unique(objs, return_counts=True))]
        bot.send_message(message.chat.id, f'On this photo i recognize: ')
        [bot.send_message(message.chat.id, f'{msg}') for msg in msgs]

        if set(cfg.needClasses) & set(objs):
            images, breeds = detector.find_needed_classes(source_img, objs, bboxes)

            for img, breed in zip(images, breeds):
                print(breed)
                processed_path = save_photo(img, path, addition=str(breed))
                bot.send_photo(message.chat.id, open(processed_path, 'rb'))
                bot.send_message(message.chat.id, f'On this photo i recognize: {breed} ')

    else:
        bot.send_message(message.chat.id, f'I can\'n recognize anything on this photo. Try another one, please')


def save_photo(img, path, addition='processed'):
    new_path = path.split('.')[0] + '_' + addition + '.' + path.split('.')[1]
    cv2.imwrite(new_path, img)

    return new_path


def get_photo_from_msg(message):
    photo_id = message.photo[-1].file_id
    file_photo = bot.get_file(photo_id)
    file_name, file_extension = os.path.splitext(file_photo.file_path)
    downloaded_photo = bot.download_file(file_photo.file_path)
    src = cfg.pathTGPhotos + photo_id + file_extension
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_photo)

    return cv2.imread(src), src


if __name__ == '__main__':
    bot.polling(none_stop=True)

