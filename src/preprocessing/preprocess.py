# src/preprocessing/preprocess.py
import pandas as pd
import re

# TEXT_PLACEHOLDER = "[NO_TEXT]"
        
# EMPTY_QUESTION_MARKERS = [
#     'nan', 'Nan', 'None', 'Null', 'пусто', 'отсутствует текст описание пустое',
#     'fwd: описание пустое', 'пустое описание', 'отсутствует текст',
#     'отсутствует', 'нету', 'нет'
# ]

# SERVICE_MARKERS = [
#     r"The mail",
#     r"Уведомляющий Сервер",
#     r"Postfix Queue ID",
#     r"Postfix Sender",
#     r"Arrival Date",
#     r"Recipient:",
#     r"Original Recipient:",
#     r"Diagnostic Code",
#     r"in reply to RCPT TO command",
#     r"Mail received:",
#     r"Message ID:",
#     r"Mail server"
#     r"Receive Notification Mail",
# ]

# SENSITIVE_PATTERNS = {
#     "EMAIL": r'\b[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b|<".+?"@[^>]+>',
#     "INN": r'(?i)инн\s*\d{5,12}',
#     "PHONE": r'\b(?:\+7|8)?[\s-]?\(?\d{3,4}\)?[\s-]?\d{2,3}[\s-]?\d{2}[\s-]?\d{2}(?:\s*\(доб\.\s*\d+\))?\b',
#     # "PHONE_EXT": r'\b8\s?\d{3,4}[\s-]?\d{2}[\s-]?\d{2}[\s-]?\d{2}(?:\s*\(доб\.\s*\d+\))?',
#     "VIN": r'\b[A-HJ-NPR-Z0-9]{17}\b',
#     "INCEDENT": r'\bIM\d{8-12}\b',
#     "REG_NUMBER": r'\b[АВЕКМНОРСТУХ]{1,3}\d{3,4}[АВЕКМНОРСТУХ]{2}\b',
#     "CASE_NO": r'\b(?:[CcSs]D|T)\d{6,10}\b',
#     "APPEAL_NO": r'\b\d/\d{9,12}\b',
#     "DOC_NO": r'№\s?\d{1,6}([/\\-]?\d{1,6})?([А-ЯA-Z])?',
#     "LONG_ID": r'\b\d{9,}\b|\b[a-f0-9]{32,128}\b|(?:UID|GlndID|GUID)[: ]?[0-9a-fA-F\-]{16,128}',
#     "IP": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
#     "DATE": r'\b\d{2}[./-]\d{2}[./-]\d{2,4}(?:\s?г\.?)?',
#     "FIO": r'\b[А-ЯЁ][а-яё]+ [А-ЯЁ]\.[А-ЯЁ]\.(?=[\s,.)/]|$)',
#     "USERNAME": r'\b[a-zA-Z][a-zA-Z0-9]{3,15}\d\b',
#     "TOKEN": r'\b[a-f0-9]{16,64}\b|\b(?=.*[A-Z])(?=.*[a-z])(?=.*\d)[A-Za-z0-9]{12,64}\b',
#     "URL": r'https?://[^\s]+',
# }
    
class Preprocessor:
    def __init__(self, cfg: dict):
        self.text_cols = cfg['text_cols']
        self.text_placeholder = cfg['TEXT_PLACEHOLDER']
        self.empty_question_markers = cfg['EMPTY_QUESTION_MARKERS']
        self.service_markers = cfg['SERVICE_MARKERS']   
        self.sensitive_patterns = cfg['SENSITIVE_PATTERNS']

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str) or len(text.strip()) < 3:
            return self.text_placeholder
        
        # remove unwanted values
        if text.lower() in [str(u).lower() for u in self.empty_question_markers]:
            return self.text_placeholder
        if len(text.split()) <= 1:
            return self.text_placeholder
        
        # remove empty question markers from the text
        pattern = r'\b(' + '|'.join(re.escape(str(item)) for item in self.empty_question_markers if not pd.isna(item)) + r')\b'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # strip service info
        pattern = re.compile("|".join(self.service_markers), re.IGNORECASE)
        match = pattern.search(text)
        if match and match.start() < 10:
            return self.text_placeholder
        elif match:
            text = text[:match.start()].rstrip()
        elif re.match(r"^[\x80-\xFF]{5,}", text):
            return self.text_placeholder 
        
        for placeholder, pattern in self.sensitive_patterns.items():
            text = re.sub(pattern, f"[{placeholder}]", text, flags=re.IGNORECASE)
        
        text = text.strip()
        text = re.sub(r"<.*?>", " ", text)  # remove HTML
        text = re.sub(r"[\n\r\t]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        
        if len(text.split()) < 3:
            return self.text_placeholder
        
        return text
    
    def clean_dataframe(self, df: pd.DataFrame):
        for col in self.text_cols:
            df[col + "_clean"] = df[col].map(self.clean_text)
        return df