"""
dataset/aux_labeler.py
======================
Heuristik Multi-Level Thinking + Confidence-Weighted Labels.

Setiap fungsi mengembalikan (label: int, confidence: float).
Confidence digunakan untuk menykalakan auxiliary loss:
    loss_aux *= confidence

Heads:
- L8  (idx 7)  : Syntax       (4-class)
- L12 (idx 11) : Concept      (6-class) — Language/Framework/Infra
- L16 (idx 15) : Semantic     (4-class)
- L24 (idx 23) : Architecture (7-class)
- L32 (idx 31) : Reasoning    (10-class)
"""

import re
import torch


# ==============================================================================
# 1. SYNTAX HEAD — Layer 8 (4-class)
# ==============================================================================

def compute_syntax_label(text: str) -> tuple[int, float]:
    """
    0 = Teks bebas / Natural Language
    1 = Deklarasi sederhana (variabel, konstanta)
    2 = Blok kontrol struktur (if, for, while, def)
    3 = Struktur data/sintaks kompleks (JSON bersarang, regex, nested dict)
    """
    # 3: Kompleks — sangat spesifik, confidence tinggi
    if re.search(r'(\[.*\[.*\]\]|\{.*\{.*\}\})', text):
        return 3, 0.90
    if re.search(r'import re|re\.compile|re\.search', text):
        return 3, 0.95

    # 2: Control flow — keyword yang sangat jelas
    t = text.lower()
    flow_count = sum(1 for kw in ['if ', 'else:', 'elif ', 'for ', 'while ', 'def ', 'class ', 'switch', 'case '] if kw in t)
    if flow_count >= 2:
        return 2, 0.95
    if flow_count == 1:
        return 2, 0.80

    # 1: Deklarasi
    if '=' in text and re.search(r'\b(var|let|const|int|float|string|self\.\w+)\b', t):
        return 1, 0.85
    if '=' in text:
        return 1, 0.60  # Banyak false positive

    # 0: Teks natural
    return 0, 0.70


# ==============================================================================
# 2. CONCEPT HEAD — Layer 12 (6-class)
# ==============================================================================

def compute_concept_label(text: str) -> tuple[int, float]:
    """
    0 = Python
    1 = JavaScript / TypeScript
    2 = Systems (C/C++/Rust/Go)
    3 = JVM (Java/Kotlin/Scala)
    4 = Infra/Config (YAML, Docker, K8s)
    5 = Natural Language (tidak ada sinyal kode)
    """
    t = text.lower()

    # Python — sangat khas
    if re.search(r'\b(def\s+\w+\(self|import\s+\w+|from\s+\w+\s+import|\.py\b|__init__|print\()', text):
        return 0, 0.90

    # JavaScript / TypeScript
    if re.search(r'\b(const\s+\w+\s*=|=>\s*\{|require\(|module\.exports|\.tsx?\b|async\s+function)\b', text):
        return 1, 0.85

    # Systems
    if re.search(r'(#include|malloc|free|unsafe\s*\{|fn\s+\w+|impl\s+\w+|go\s+func|chan\s+\w+)', text):
        return 2, 0.90

    # JVM
    if re.search(r'\b(public\s+class|private\s+\w+|@Override|System\.out|fun\s+\w+\(|val\s+\w+\s*=)', text):
        return 3, 0.85

    # Infra/Config
    if re.search(r'(apiVersion:|FROM\s+\w+|services:|kind:\s*Deployment|image:\s)', text):
        return 4, 0.95

    # Natural Language — default, low confidence karena bisa juga bahasa langka
    return 5, 0.50


# ==============================================================================
# 3. SEMANTIC HEAD — Layer 16 (4-class)
# ==============================================================================

def compute_semantic_label(text: str) -> tuple[int, float]:
    """
    0 = Netral / Tidak ada komputasi
    1 = Manipulasi string / I/O / Data processing
    2 = Algoritma, struktur data matematis (rekursi, sorting)
    3 = Konsep keamanan dan logik bisnis kompleks (enkripsi, auth)
    """
    t = text.lower()

    # 3: Security & Business Logic
    if re.search(r'\b(encrypt|decrypt|hash|hmac|jwt|oauth|ssl|tls|vulnerability|exploit|cve-)\b', t):
        return 3, 0.92

    # 2: Algorithms
    if re.search(r'\b(sort|search|tree|graph|node|recursion|recursive|math\.|numpy|tensor|matrix)\b', t):
        return 2, 0.80

    # 1: Basic I/O
    if re.search(r'\b(print|console\.log|read|write|open\(|close\(|split|join|replace|format)\b', t):
        return 1, 0.75

    # 0: Netral
    return 0, 0.65


# ==============================================================================
# 4. ARCHITECTURE HEAD — Layer 24 (7-class)
# ==============================================================================

def compute_architecture_label(text: str) -> tuple[int, float]:
    """
    0 = Standalone (script lepas)
    1 = MVC / MVVM
    2 = Clean / Hexagonal
    3 = Repository Pattern
    4 = Microservice
    5 = Event-Driven
    6 = API / REST / GraphQL
    """
    t = text.lower()

    # 6: API / REST / GraphQL — sangat umum, confidence sedang
    if re.search(r'\b(app\.get|app\.post|@app\.route|endpoint|@GetMapping|router\.|resolver|mutation|graphql)\b', t):
        return 6, 0.85

    # 5: Event-Driven
    if re.search(r'\b(event.?emitter|publish|subscribe|on\(|emit\(|kafka|rabbitmq|message.?queue|nats)\b', t):
        return 5, 0.88

    # 4: Microservice
    if re.search(r'\b(microservice|service.?mesh|gateway|consul|istio|load.?balancer|grpc|proto(buf)?)\b', t):
        return 4, 0.85

    # 3: Repository Pattern
    if re.search(r'\b(repository|dao|find.?by.?id|find.?all|save\(|delete.?by|data.?source)\b', t):
        return 3, 0.80

    # 2: Clean / Hexagonal
    if re.search(r'\b(use.?case|interactor|port|adapter|domain.?model|hexagonal|clean.?arch|bounded.?context)\b', t):
        return 2, 0.75  # Sulit dideteksi hanya dari teks

    # 1: MVC / MVVM
    if re.search(r'\b(controller|view.?model|mvvm|mvc|@controller|@service|template|render)\b', t):
        return 1, 0.70  # Keyword sangat generik

    # 0: Standalone
    return 0, 0.60


# ==============================================================================
# 5. REASONING HEAD — Layer 32 (10-class)
# ==============================================================================

def compute_reason_label(text: str) -> tuple[int, float]:
    """
    0  = None (teks biasa)
    1  = Data Flow (variabel, assignment chain)
    2  = Control Flow (branching, state machine)
    3  = Authentication (login, session, JWT)
    4  = Concurrency (thread, async, lock)
    5  = Memory (allocation, pointer, buffer)
    6  = Protocol (HTTP, TCP, WebSocket)
    7  = Parsing (AST, regex engine, tokenizer)
    8  = Optimization (caching, Big-O, profiling)
    9  = Security Reasoning (exploit chain, CVE analysis)
    """
    t = text.lower()

    # 9: Security Reasoning — paling kritis
    if re.search(r'\b(exploit|shellcode|reverse.shell|payload|cve-\d|privilege.escalation|rce|injection)\b', t):
        return 9, 0.95

    # 8: Optimization
    if re.search(r'\b(cache|memoiz|big.?o|o\(n|o\(log|profil|benchmark|vectoriz|simd|optimization)\b', t):
        return 8, 0.82

    # 7: Parsing
    if re.search(r'\b(parser|ast|tokenizer|lexer|grammar|regex|bnf|syntax.tree|recursive.descent)\b', t):
        return 7, 0.90

    # 6: Protocol
    if re.search(r'\b(http|tcp|udp|websocket|grpc|protobuf|rest|api.call|request|response|status.code)\b', t):
        return 6, 0.75  # Terlalu umum

    # 5: Memory
    if re.search(r'\b(malloc|free|heap|stack|buffer|pointer|segfault|memory.leak|garbage|mmap|alloc)\b', t):
        return 5, 0.92

    # 4: Concurrency
    if re.search(r'\b(thread|mutex|semaphore|deadlock|race.condition|async|await|lock|goroutine|channel)\b', t):
        return 4, 0.88

    # 3: Authentication
    if re.search(r'\b(login|logout|session|jwt|oauth|bearer|password|credential|auth|permission)\b', t):
        return 3, 0.80

    # 2: Control Flow
    cflow = sum(1 for kw in ['if ', 'else', 'elif', 'switch', 'case ', 'while ', 'for '] if kw in t)
    if cflow >= 3:
        return 2, 0.90
    if cflow >= 1:
        return 2, 0.65

    # 1: Data Flow
    if '=' in text and re.search(r'\w+\s*=\s*\w+', text):
        return 1, 0.55  # Sangat generik

    # 0: None
    return 0, 0.70


# ==============================================================================
# BATCH GENERATOR
# ==============================================================================

def compute_aux_labels_batch(texts: list[str], device: torch.device) -> tuple[dict, dict]:
    """
    Hitung semua label + confidence untuk satu batch teks.

    Returns:
        (labels_dict, confidences_dict)
        labels_dict: {'syntax': LongTensor[B], 'concept': ..., ...}
        confidences_dict: {'syntax': FloatTensor[B], 'concept': ..., ...}
    """
    syn_l, syn_c = [], []
    con_l, con_c = [], []
    sem_l, sem_c = [], []
    arc_l, arc_c = [], []
    rea_l, rea_c = [], []

    for text in texts:
        l, c = compute_syntax_label(text)
        syn_l.append(l); syn_c.append(c)

        l, c = compute_concept_label(text)
        con_l.append(l); con_c.append(c)

        l, c = compute_semantic_label(text)
        sem_l.append(l); sem_c.append(c)

        l, c = compute_architecture_label(text)
        arc_l.append(l); arc_c.append(c)

        l, c = compute_reason_label(text)
        rea_l.append(l); rea_c.append(c)

    labels = {
        'syntax':       torch.tensor(syn_l, dtype=torch.long, device=device),
        'concept':      torch.tensor(con_l, dtype=torch.long, device=device),
        'semantic':     torch.tensor(sem_l, dtype=torch.long, device=device),
        'architecture': torch.tensor(arc_l, dtype=torch.long, device=device),
        'reasoning':    torch.tensor(rea_l, dtype=torch.long, device=device),
    }
    confidences = {
        'syntax':       torch.tensor(syn_c, dtype=torch.float, device=device),
        'concept':      torch.tensor(con_c, dtype=torch.float, device=device),
        'semantic':     torch.tensor(sem_c, dtype=torch.float, device=device),
        'architecture': torch.tensor(arc_c, dtype=torch.float, device=device),
        'reasoning':    torch.tensor(rea_c, dtype=torch.float, device=device),
    }
    return labels, confidences


if __name__ == "__main__":
    samples = [
        "def scan_port(host, port):\n    if port == 443:\n        return True",
        "const app = require('express'); app.get('/', (req, res) => res.send('hi'))",
        "The CVE-2024-1234 exploit uses a buffer overflow to achieve RCE.",
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: sentinel",
    ]
    device = torch.device('cpu')
    labels, confs = compute_aux_labels_batch(samples, device)
    for i, t in enumerate(samples):
        print(f"\nSample {i+1}: {t[:60].replace(chr(10), ' ')}...")
        for key in labels:
            print(f"  {key:>15}: label={labels[key][i].item()}, conf={confs[key][i].item():.2f}")
