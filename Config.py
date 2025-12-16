from bdrc.data import (
    CharsetEncoder,
    Encoding,
    ExportFormat,
    Language,
    LineMerge,
    LineMode,
    LineSorting,
    OCRArchitecture,
    TPSMode,
)

"""
MODELS on HuggingFace
"""

MODEL_DICT = {
    "Lines_v1": "BDRC/PhotiLines",
    "Photi_v2": "BDRC/Photi-v2",
    "Woodblock": "BDRC/Woodblock",
    "UCHAN": "BDRC/BigUCHAN_v1",
    "DergeTenjur": "BDRC/DergeTenjur",
    "GoogleBooks_C": "BDRC/GoogleBooks_C_v1",
    "GoogleBooks_E": "BDRC/GoogleBooks_E_v1",
    "Norbuketaka_C": "BDRC/Norbuketaka_C_V1",
    "Norbuketaka_E": "BDRC/Norbuketaka_E_V1",
    "Drutsa-A_E": "BDRC/Drutsa-A_E_v1",
}


"""
Mappings for each data type
"""

COLOR_DICT = {
    "background": "0, 0, 0",
    "image": "45, 255, 0",
    "text": "255, 243, 0",
    "margin": "0, 0, 255",
    "caption": "255, 100, 243",
    "table": "0, 255, 0",
    "pagenr": "0, 100, 15",
    "header": "255, 0, 0",
    "footer": "255, 255, 100",
    "line": "0, 100, 255",
}
"""
page_classes = {
    "background": "0, 0, 0",
    "image": "45, 255, 0",
    "line": "255, 100, 0",
    "margin": "255, 0, 0",
    "caption": "255, 100, 243",
}"""


LANGUAGES = {
    "en": Language.ENGLISH,
    "de": Language.GERMAN,
    "fr": Language.FRENCH,
    "bo": Language.TIBETAN,
    "ch": Language.CHINESE,
}

ENCODINGS = {"unicode": Encoding.UNICODE, "wylie": Encoding.WYLIE}

CHARSETENCODER = {"wylie": CharsetEncoder.WYLIE, "stack": CharsetEncoder.STACK}

OCRARCHITECTURE = {"Easter2": OCRArchitecture.EASTER2, "CRNN": OCRArchitecture.CRNN}

EXPORTERS = {"xml": ExportFormat.XML, "json": ExportFormat.JSON, "text": ExportFormat.TXT}

LINE_MODES = {"line": LineMode.LINE, "layout": LineMode.LAYOUT}

LINE_MERGE = {"merge": LineMerge.MERGE, "stack": LineMerge.STACK}

LINE_SORTING = {"threshold": LineSorting.THRESHOLD, "peaks": LineSorting.PEAKS}

TPS_MODE = {"local": TPSMode.LOCAL, "global": TPSMode.GLOBAL}
