import logging
import cv2
import os
from aiogram import Bot, Dispatcher, executor, types
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf


#включаем логирование
logging.basicConfig(level=logging.INFO)
#объект бота
API_TOKEN = '5966372854:AAH5PcV4rOSMPG5YMzy2P49xwzKoUyjrWk8'
bot = Bot(token=API_TOKEN)
#диспетчер
dp = Dispatcher(bot)
#обработка команды старт
@dp.message_handler(commands=['start'])
async def echo(message: types.Message):
  await message.reply('Привет! Я бот, распознающий грибы')
# обработка команды help
@dp.message_handler(commands=['help'])
async def echo(message: types.Message):
  await message.reply('Просто отправьте мне изображение, которое содержит гриб')

@dp.message_handler(content_types=[types.ContentType.PHOTO])
async def download_photo(message: types.Message):
# загружаем фото в папку по умолчанию
   await message.photo[-1].download()
# определяем путь к фото
   img_path = (await bot.get_file(message.photo[-1].file_id)).file_path
# получаем предсказание
   pred = predictions(img_path)
# Отправляем ответ пользователю
   await message.answer(f"Я думаю, что это {pred}")
def get_img_array(img_path, size):
  img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
  array = tf.keras.preprocessing.image.img_to_array(img)
  # расширяем размерность для преобразования массива в пакеты
  array = np.expand_dims(array, axis=0)
  return array


def predictions(img_path):
  img_size = (224, 224, 3)
  classifier = load_model("C:/Users/Admin/PycharmProjects/Mushroom/mushroom13mobile.h5")
  class_labels = ['conditionally_edible-Amanita_fulva', 'conditionally_edible-Amanita_rubescens', 'conditionally_edible-Coprinopsis_atramentaria',
                  'conditionally_edible-Lactarius_torminosus', 'conditionally_edible-Lepista_saeva', 'conditionally_edible-Morchella_esculenta',
                  'conditionally_edible-Pleurocybella_porrigens', 'conditionally_edible-Verpa_bohemica', 'deadly-Amanita_arocheae',
                  'deadly-Amanita_bisporigera', 'deadly-Amanita_exitialis', 'deadly-Amanita_fuliginea', 'deadly-Amanita_magnivelaris',
                  'deadly-Amanita_muscaria', 'deadly-Amanita_ocreata', 'deadly-Amanita_phalloides', 'deadly-Amanita_smithiana',
                  'deadly-Amanita_sphaerobulbosa', 'deadly-Amanita_subjunquillea', 'deadly-Amanita_subpallidorosea', 'deadly-Amanita_verna',
                  'deadly-Amanita_virosa', 'deadly-Clitocybe_dealbata', 'deadly-Clitocybe_rivulosa', 'deadly-Cortinarius_eartoxicus',
                  'deadly-Cortinarius_orellanus', 'deadly-Cortinarius_rubellus', 'deadly-Cortinarius_splendens', 'deadly-Entoloma_sinuatum',
                  'deadly-Galerina_marginata', 'deadly-Galerina_sulciceps', 'deadly-Hypholoma_fasciculare', 'deadly-Inosperma_erubescens',
                  'deadly-Lepiota_brunneoincarnata', 'deadly-Lepiota_brunneolilacea', 'deadly-Lepiota_castanea', 'deadly-Lepiota_helveola',
                  'deadly-Lepiota_subincarnata', 'deadly-Omphalotus_illudens', 'deadly-Paxillus_involutus', 'deadly-Pholiotina_rugosa',
                  'deadly-Podostroma_cornu-damae', 'deadly-Russula_subnigricans', 'deadly-Tricholoma_equestre', 'deadly-Trogia_venenata',
                  'edible-Agaricus_arvensis', 'edible-Agaricus_bisporus', 'edible-Agaricus_silvaticus', 'edible-Aleuria_aurantia',
                  'edible-Amanita_caesarea', 'edible-Armillaria_mellea', 'edible-Auricularia_auricula-judae', 'edible-Boletus_badius',
                  'edible-Boletus_edulis', 'edible-Calbovista_subsculpta', 'edible-Calocybe_gambosa', 'edible-Calvatia_gigantea',
                  'edible-Calvatia_utriformis', 'edible-Cantharellus_cibarius', 'edible-Chroogomphus', 'edible-Clavariaceae', 'edible-Clavulinaceae',
                  'edible-Clitocybe_nuda', 'edible-Coprinus_comatus', 'edible-Cortinarius_caperatus', 'edible-Cortinarius_variicolor',
                  'edible-Craterellus_cornucopioides', 'edible-Craterellus_tubaeformis', 'edible-Cyclocybe_aegerita', 'edible-Cyttaria_espinosae',
                  'edible-Fistulina_hepatica', 'edible-Flammulina_velutipes', 'edible-Grifola_frondosa', 'edible-Hericium_erinaceus',
                  'edible-Hydnum_repandum', 'edible-Hygrophorus_chrysodon', 'edible-Hypsizygus_tessellatus', 'edible-Kalaharituber_pfeilii',
                  'edible-Lactarius_deliciosus', 'edible-Lactarius_deterrimus', 'edible-Lactarius_subdulcis', 'edible-Lactarius_volemus',
                  'edible-Laetiporus_sulphureus', 'edible-Leccinum_aurantiacum', 'edible-Leccinum_scabrum', 'edible-Leccinum_versipelle',
                  'edible-Lentinula_edodes', 'edible-Macrolepiota_procera', 'edible-Marasmius_oreades', 'edible-Phallus_indusiatus',
                  'edible-Pleurotus', 'edible-Polyporus_mylittae', 'edible-Polyporus_squamosus', 'edible-Pseudohydnum_gelatinosum',
                  'edible-Rhizopogon_luteolus', 'edible-Sparassis_crispa', 'edible-Stropharia_rugosoannulata', 'edible-Suillus_bovinus',
                  'edible-Suillus_granulatus', 'edible-Suillus_luteus', 'edible-Suillus_tomentosus', 'edible-Tremella_fuciformis',
                  'edible-Tricholoma_matsutake', 'edible-Tricholoma_terreum', 'edible-Tuber_aestivum', 'edible-Tuber_borchii', 'edible-Tuber_brumale',
                  'edible-Tuber_indicum', 'edible-Tuber_macrosporum', 'edible-Tuber_mesentericum', 'edible-Volvariella_volvacea',
                  'poisonous-Boletus_pulcherrimus', 'poisonous-Calocera_viscosa', 'poisonous-Chlorophyllum_brunneum', 'poisonous-Chlorophyllum_molybdites',
                  'poisonous-Choiromyces_venosus', 'poisonous-Clitocybe_cerussata', 'poisonous-Clitocybe_dealbata', 'poisonous-Clitocybe_fragrans',
                  'poisonous-Clitocybe_nebularis', 'poisonous-Conocybe_subovalis', 'poisonous-Coprinellus_micaceus', 'poisonous-Coprinopsis_alopecia',
                  'poisonous-Coprinopsis_atramentaria', 'poisonous-Coprinopsis_romagnesiana', 'poisonous-Cortinarius_bolaris',
                  'poisonous-Cortinarius_callisteus', 'poisonous-Cortinarius_cinnabarinus', 'poisonous-Cortinarius_cinnamomeoluteus',
                  'poisonous-Cortinarius_cinnamomeus', 'poisonous-Cortinarius_cruentus', 'poisonous-Cortinarius_limonius', 'poisonous-Cortinarius_malicorius',
                  'poisonous-Cortinarius_mirandus', 'poisonous-Cortinarius_palustris', 'poisonous-Cortinarius_phoeniceus', 'poisonous-Cortinarius_rubicundulus',
                  'poisonous-Cortinarius_smithii', 'poisonous-Cudonia_circinans', 'poisonous-Echinoderma_asperum', 'poisonous-Echinoderma_calcicola',
                  'poisonous-Entoloma_albidum', 'poisonous-Entoloma_rhodopolium', 'poisonous-Entoloma_sinuatum', 'poisonous-Gyromitra_esculenta',
                  'poisonous-Gyromitra_perlata', 'poisonous-Hapalopilus_nidulans', 'poisonous-Hebeloma_crustuliniforme', 'poisonous-Hebeloma_sinapizans',
                  'poisonous-Helvella_crispa', 'poisonous-Helvella_dryophila', 'poisonous-Helvella_lactea', 'poisonous-Helvella_lacunosa',
                  'poisonous-Helvella_vespertina', 'poisonous-Hypholoma_fasciculare', 'poisonous-Hypholoma_lateritium', 'poisonous-Hypholoma_marginatum',
                  'poisonous-Hypholoma_radicosum', 'poisonous-Imperator_rhodopurpureus', 'poisonous-Imperator_torosus', 'poisonous-Inocybe_fibrosa',
                  'poisonous-Inocybe_geophylla', 'poisonous-Inocybe_sambucina', 'poisonous-Inocybe_sublilacina', 'poisonous-Lactarius_chrysorrheus',
                  'poisonous-Lactarius_helvus', 'poisonous-Lactarius_torminosus', 'poisonous-Lepiota_cristata', 'poisonous-Marasmius_collinus',
                  'poisonous-Mycena_diosma', 'poisonous-Mycena_pura', 'poisonous-Mycena_rosea', 'poisonous-Neonothopanus_nambi', 'poisonous-Omphalotus_illudens',
                  'poisonous-Omphalotus_japonicus', 'poisonous-Omphalotus_nidiformis', 'poisonous-Omphalotus_olearius', 'poisonous-Omphalotus_olivascens',
                  'poisonous-Panaeolus_cinctulus', 'poisonous-Paralepistopsis_amoenolens', 'poisonous-Pholiotina_rugosa', 'poisonous-Psilocybe_semilanceata',
                  'poisonous-Ramaria_formosa', 'poisonous-Ramaria_neoformosa', 'poisonous-Ramaria_pallida', 'poisonous-Rubroboletus_legaliae',
                  'poisonous-Rubroboletus_lupinus', 'poisonous-Rubroboletus_pulcherrimus', 'poisonous-Rubroboletus_satanas', 'poisonous-Scleroderma_citrinum',
                  'poisonous-Stropharia_aeruginosa', 'poisonous-Suillus_granulatus', 'poisonous-Tricholoma_equestre', 'poisonous-Tricholoma_filamentosum',
                  'poisonous-Tricholoma_muscarium', 'poisonous-Tricholoma_pardinum', 'poisonous-Tricholoma_sulphureum', 'poisonous-Trogia_venenata',
                  'poisonous-Turbinellus_floccosus', 'poisonous-Turbinellus_kauffmanii']
  try:
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    pred = np.argmax(classifier.predict(img_array), axis=1)
    predictions = class_labels[pred[0]]
    return predictions
  except Exception:
    return "Проблемы с изображением"

if __name__ == '__main__':
#запуск пуллинга
  executor.start_polling(dp, skip_updates=True)