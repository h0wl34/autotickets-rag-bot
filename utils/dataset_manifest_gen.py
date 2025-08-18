#!/usr/bin/env python3
"""
Генератор метаданных для датасетов

Примеры использования:
1. Базовый вариант:
   python dataset_metadata.py --input cleaned_dataset.csv --output metadata.json

2. С дополнительными заметками:
   python dataset_metadata.py -i data.csv -o meta.json -n "preprocessed version"

3. Подробный режим (со статистикой):
   python dataset_metadata.py -i data.csv --stats
"""

import argparse
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

def generate_metadata(df: pd.DataFrame, filename: str, notes: str = "", include_stats: bool = False) -> dict:
    """Генерирует метаданные датасета."""
    metadata = {
        "file": filename,
        "rows": len(df),
        "columns": list(df.columns),
        "created_at": datetime.now().isoformat(),
        "notes": notes,
    }
    
    if include_stats:
        metadata["stats"] = {
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_values": df.isnull().sum().to_dict(),
            "unique_values": df.nunique().to_dict()
        }
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description="Генератор метаданных для датасетов")
    parser.add_argument("-i", "--input", required=True, help="Входной CSV файл с данными")
    parser.add_argument("-o", "--output", help="Выходной JSON файл")
    parser.add_argument("-n", "--notes", default="", help="Заметки о преобразованиях данных")
    parser.add_argument("--stats", action="store_true", help="Включить статистику по колонкам")
    parser.add_argument("--encoding", default="utf-8", help="Кодировка файлов")
    
    args = parser.parse_args()
    
    try:
        # Чтение данных
        df = pd.read_csv(args.input, encoding=args.encoding)
        print(f"✓ Данные прочитаны: {len(df)} строк, {len(df.columns)} колонок")
        
        # Генерация метаданных
        metadata = generate_metadata(
            df, 
            filename=Path(args.input).name,
            notes=args.notes,
            include_stats=args.stats
        )
        
        # Сохранение
        if (args.output):
            out_path = Path(args.output)
        else:
            input_dir = Path(args.input).parent
            out_path = input_dir / f"manifest_{datetime.now().strftime('%Y%m%d')}.json"

        with open(out_path, "w", encoding=args.encoding) as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"✓ Метаданные сохранены в {out_path}")
        sys.exit(0)
        
    except Exception as e:
        print(f"✗ Ошибка: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()