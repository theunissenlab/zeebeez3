import os

def get_root_dir():
    dirname, filename = os.path.split(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(dirname, '..'))

def get_data_dir():
    root_dir = get_root_dir()
    return os.path.join(root_dir, 'data')

CALL_TYPES = ['Be', 'LT', 'Ne', 'Te', 'DC', 'Ag', 'Di', 'Th', 'song']

# DECODER_CALL_TYPES = ['Be', 'LT', 'Ne', 'Te', 'DC', 'Ag', 'song']
DECODER_CALL_TYPES = ['Be', 'LT', 'Ne', 'Te', 'DC', 'Ag', 'Di', 'Th', 'song']

CALL_TYPE_COLORS = {'Ag':'r', 'DC':'#6600CC', 'Ne':'#FF33CC', 'Te':'#9966CC', 'Di':'#FF6600', 'song':'#606060',
                    'Th':'#FF9966', 'Be':'#00FFCC', 'LT':'#0066FF', 'mlnoise':'#33FF33'}

CALL_TYPE_NAMES = {'Ag': 'Wsst', 'Be': 'Begging', 'DC': 'Distance Call', 'Di': 'Distress', 'LT': 'Long Tonal',
                   'Ne': 'Nest', 'Te': 'Tet', 'Th': 'Thuck', 'mlnoise': 'ML Noise', 'song': 'Song'}

CALL_TYPE_SHORT_NAMES = {'Ag': 'Ws', 'Be': 'Be', 'DC': 'DC', 'Di': 'Di', 'LT': 'LT',
                         'Ne': 'Ne', 'Te': 'Te', 'Th': 'Th', 'mlnoise': 'ML', 'song': 'So'}

ACOUSTIC_FEATURE_COLORS = {'voice2percent':'#944D0E',
                           'sal':'#F0DB00',
                           'fund':'#006E41',
                           'maxAmp':'k',
                           'meanspect':'#E96652',
                           'skewspect':'#E31590',
                           'entropyspect':'#EFB2C9',
                           'q2':'#E90027',
                           'meantime':'#002E84',
                           'skewtime':'#257CDF',
                           'entropytime':'#8231A5',
                          }

ACOUSTIC_FUND_PROPS = ['minfund', 'maxfund', 'fund', 'fund2', 'cvfund', 'voice2percent']

REGION_COLORS = {'CMM':'#D60036', 'CML':'#CC00CC', 'L1':'#290088', 'L2':'#26CAD3', 'L3':'#0055B8', 'NCM':'#EBD81F'}

REGION_COLORS_SHORT = {'CM':'#D60036', 'L1':'#290088', 'L2':'#26CAD3', 'L3':'#0055B8', 'NCM':'#EBD81F'}
REGION_NAMES_SHORT = ['CM', 'L1', 'L2', 'L3', 'NCM']

REGION_NAMES = ['CMM', 'CML', 'L1', 'L2', 'L3', 'NCM']

REGION_NAMES_LONG = ['CMM', 'CML', 'L1', 'L2A', 'L2B', 'L3', 'NCM']

ROSTRAL_CAUDAL_ELECTRODES_LEFT = [1, 16, 2, 15, 3, 14, 4, 13, 5, 12, 6, 11, 7, 10, 8, 9]

ROSTRAL_CAUDAL_ELECTRODES_RIGHT = [24, 25, 23, 26, 22, 27, 21, 28, 20, 29, 19, 30, 18, 31, 17, 32]

REDUCED_ACOUSTIC_PROPS = ('fund', 'sal', 'voice2percent', 'q2', 'meantime', 'entropytime', 'maxAmp')

DISPLAYED_ACOUSTIC_PROPS = ['sal', 'meanspect', 'entropyspect', 'q2', 'maxAmp', 'skewtime']

USED_ACOUSTIC_PROPS = ['fund', 'fund2', 'sal', 'voice2percent', 'maxfund', 'minfund', 'cvfund',
                       'meanspect', 'stdspect', 'skewspect', 'kurtosisspect', 'entropyspect',
                       'q1', 'q2', 'q3', 'stdtime', 'skewtime', 'kurtosistime', 'entropytime', 'maxAmp']

ACOUSTIC_PROP_NAMES = {'fund':'Mean F0', 'fund2':'Pk 2', 'sal':'Saliency', 'voice2percent':'2nd V',
                       'maxfund':'Max F0', 'minfund':'Min F0', 'cvfund':'CV F0',
                       'meanspect':'Mean S', 'stdspect':'Std S', 'skewspect':'Skew S', 'kurtosisspect':'Kurt S',
                       'entropyspect':'Ent S', 'q1':'Q1', 'q2':'Q2', 'q3':'Q3',
                       'stdtime':'Std T', 'skewtime':'Skew T', 'kurtosistime':'Kurt T', 'entropytime':'Ent T',
                       'maxAmp':'Max A'
                      }

ACOUSTIC_PROP_FULLNAMES = {'fund':'Mean Fundamental', 'fund2':'2nd Fundamental', 'sal':'Saliency', 'voice2percent':'2nd Voice',
                           'maxfund':'Max Fundamental', 'minfund':'Min Fundamental', 'cvfund':'CV Fundamental',
                           'meanspect':'Mean Spectral Freq', 'stdspect':'Spectral Freq SD', 'skewspect':'Spectral Freq Skew', 'kurtosisspect':'Spectral Freq Kurtosis',
                           'entropyspect':'Spectral Entropy', 'q1':'Spectral Q1', 'q2':'Spectral Q2', 'q3':'Spectral Q3',
                           'stdtime':'Tempora SD', 'skewtime':'Temporal Skew', 'kurtosistime':'Temporal Kurtosis', 'entropytime':'Temporal Entropy',
                           'maxAmp':'Max Amplitude'
                          }

APROP_GREEN = (0., 153., 0.)
APROP_ORANGE = (255., 128., 0.)
APROP_BLUE = (0., 128., 255.)
APROP_BLACK = (0., 0., 0.)

ACOUSTIC_PROP_COLORS_BY_TYPE = {'fund':APROP_GREEN, 'fund2':APROP_GREEN, 'sal':APROP_GREEN, 'voice2percent':APROP_GREEN,
                                'maxfund':APROP_GREEN, 'minfund':APROP_GREEN, 'cvfund':APROP_GREEN,
                                'meanspect':APROP_ORANGE, 'stdspect':APROP_ORANGE, 'skewspect':APROP_ORANGE, 'kurtosisspect':APROP_ORANGE, 'entropyspect':APROP_ORANGE,
                                'q1':APROP_ORANGE, 'q2':APROP_ORANGE, 'q3':APROP_ORANGE,
                                'meantime':APROP_BLUE, 'stdtime':APROP_BLUE, 'skewtime':APROP_BLUE, 'kurtosistime':APROP_BLUE, 'entropytime':APROP_BLUE,
                                'maxAmp':APROP_BLACK}


def clean_region(reg):
    if '-' in reg:
        return '?'
    if reg.startswith('L2'):
        return 'L2'
    if reg.startswith('CM'):
        return 'CM'
    return reg
