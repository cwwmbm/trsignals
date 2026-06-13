import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*auto_adjust.*')
warnings.filterwarnings('ignore', message='.*T.*is deprecated.*')
warnings.filterwarnings('ignore', message='.*Setting an item of incompatible dtype.*')
