import os
from collections import defaultdict

EXT_TO_LANG = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".js": "JavaScript",
    ".md": "Markdown",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".html": "HTML",
    ".txt": "Text",
    ".rst": "ReStructuredText"
}

def analyze_languages(corpus_dir: str):
    lang_count = defaultdict(int)
    total_files = 0
    
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue
                
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("# --- FILE:"):
                        # Contoh: # --- FILE: logger.py ---
                        filename = line.strip().split(" ")[3]
                        ext = os.path.splitext(filename)[1].lower()
                        lang = EXT_TO_LANG.get(ext, "Other")
                        lang_count[lang] += 1
                        total_files += 1
                        
    if total_files == 0:
        return {}
        
    stats = {}
    for lang, count in sorted(lang_count.items(), key=lambda x: x[1], reverse=True):
        stats[lang] = {
            "count": count,
            "percentage": (count / total_files) * 100
        }
    return stats
