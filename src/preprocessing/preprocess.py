import pandas as pd
import re

    
class Preprocessor:
    def __init__(self, cfg: dict):
        self.text_cols = cfg['text_cols']
        self.text_placeholder = cfg['TEXT_PLACEHOLDER']
        self.empty_question_markers = cfg['EMPTY_QUESTION_PATTERNS']
        self.service_kill_markers = cfg['SERVICE_KILL_MARKERS']   
        self.service_left_markers = cfg['SERVICE_LEFT_MARKERS']   
        self.sensitive_patterns = cfg['SENSITIVE_PATTERNS']
        
    def _strip_service_info(self, text):
        # left-strip markers
        while True:
            t_strip = text
            for pat in self.service_left_markers:
                match = re.match(pat, text, re.IGNORECASE)
                if match:
                    text = text[match.end():].strip()
                    break
            if text == t_strip:
                break  # no more left-strip markers

        # Kill markers
        for pat in self.service_kill_markers:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                if match.start() < 10:
                    return self.text_placeholder
                text = text[:match.start()].rstrip()
        
        return text

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str) or len(text.strip()) < 2:
            return self.text_placeholder
        
        text = text.strip()

        # remove garbage symbols
        if re.match(r"^[\x80-\xFF]{5,}", text):
            return self.text_placeholder 
        
        # remove empty question markers from the text
        for pat in self.empty_question_markers:
            pat = str(pat).lower()
            # start
            match = re.match(pat, text.lower())
            if match:
                rest = text[match.end():].strip()
                text = rest if rest else self.text_placeholder
                break
            # end
            match = re.search(pat + r'\s*$', text.lower())
            if match:
                rest = text[:match.start()].rstrip()
                text = rest if rest else self.text_placeholder
                break
        
        text = self._strip_service_info(text)

        # mask sensetive data
        for placeholder, pattern in self.sensitive_patterns.items():
            text = re.sub(pattern, f"[{placeholder}]", text, flags=re.IGNORECASE)
        
        # remove html / normalize whitespace
        text = re.sub(r"<.*?>", " ", text)  # remove HTML
        text = re.sub(r"[\n\r\t]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        
        # min words count 
        if len(text.split()) < 2:
            return self.text_placeholder
        
        return text
    
    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized cleaning for a dataframe."""
        for col in self.text_cols:
            df[col + "_clean"] = df[col].apply(self._clean_text)
        return df
    
    def transform_text(self, text: str) -> str:
        """Clean a single text string."""
        return self._clean_text(text)