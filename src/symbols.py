import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

class Symbols():
    ASCII = 'ABCDEFGHIJKLMNOPQRSTUVWYXZabcdefghijklmnopqrstuvwxyz0123456789!\,-.:;? '
    English = ASCII
    German =  ASCII + 'äöüßÄÖÜ'
    Italian = ASCII + 'àéèìíîòóùúÀÉÈÌÍÎÒÓÙÚ'
    Spanish = ASCII + '¡¿ñáéíóúÁÉÍÓÚÑ'
    French = ASCII + 'àâæçéèêëîïôœùûüÿŸÜÛÙŒÔÏÎËÊÈÉÇÆÂÀ'
    punctuation = ';:,.!?¡¿—…"«»“” []'

    ALL_GRAPHEME_CHARACTERS = list(set(ASCII + English + German + Italian + Spanish + French + punctuation))
    ALL_PHONEME_SYMBOLS = list(set(pd.read_csv(f'{current_dir}/phoneme_vocab.csv', index_col=None)['ipa']))