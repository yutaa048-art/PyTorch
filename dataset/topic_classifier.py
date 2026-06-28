"""
dataset/topic_classifier.py
============================
Regex-based topic classifier ringan untuk Knowledge Replay Buffer (KRB).
Mendeteksi ~20 topik kunci dari teks mentah.
Setiap teks bisa memiliki 0 atau 1 topik utama (greedy first-match).

Digunakan oleh trainer.py untuk memasukkan sequence ke slot topik yang tepat.
"""

import re

# Daftar topik dan regex-nya (urutan = prioritas)
TOPICS = [
    ("transformer",        re.compile(r'\b(transformer|self[_-]?attention|multi[_-]?head|positional.encoding|embeddings?\s+layer)\b', re.I)),
    ("llm",                re.compile(r'\b(language.model|llm|gpt|bert|llama|tokenizer|fine[_-]?tun|prompt.engineering|rlhf)\b', re.I)),
    ("compiler",           re.compile(r'\b(compiler|lexer|codegen|intermediate.representation|abstract.syntax.tree|ast|bytecode)\b', re.I)),
    ("parser",             re.compile(r'\b(parser|parsing|bnf|grammar|yacc|antlr|recursive.descent|shift.reduce)\b', re.I)),
    ("database",           re.compile(r'\b(sql|nosql|mongodb|postgres|mysql|redis|index|query.plan)\b|\bselect\s+|\binsert\s+into\b|\bupdate\s+\w+\s+set\b|\bdelete\s+from\b', re.I)),
    ("networking",         re.compile(r'\b(tcp|udp|socket|http[s2]?|dns|ip.address|packet|latency|bandwidth|proxy|load.balancer)\b', re.I)),
    ("jwt",                re.compile(r'\b(jwt|json.web.token|bearer|claims|hs256|rs256|refresh.token)\b', re.I)),
    ("oauth",              re.compile(r'\b(oauth|openid|authorization.code|client.credentials|scope|redirect.uri)\b', re.I)),
    ("cryptography",       re.compile(r'\b(aes|rsa|sha[_-]?256|hmac|encrypt|decrypt|cipher|public.key|private.key|certificate|tls|ssl)\b', re.I)),
    ("concurrency",        re.compile(r'\b(thread|mutex|semaphore|deadlock|race.condition|async|await|coroutine|channel|goroutine|lock)\b', re.I)),
    ("memory",             re.compile(r'\b(malloc|free|heap|stack|buffer.overflow|memory.leak|garbage.collect|pointer|segfault|mmap)\b', re.I)),
    ("scheduler",          re.compile(r'\b(scheduler|cron|task.queue|celery|job|worker|priority.queue|round.robin|preempt)\b', re.I)),
    ("container",          re.compile(r'\b(docker|dockerfile|container|image|volume|docker.compose|buildkit|registry)\b', re.I)),
    ("kubernetes",         re.compile(r'\b(kubernetes|k8s|kubectl|pod|deployment|service|ingress|helm|kube|namespace)\b', re.I)),
    ("inference",          re.compile(r'\b(inference|predict|onnx|tensorrt|quantiz|pruning|distill|serving|batch.infer)\b', re.I)),
    ("training",           re.compile(r'\b(training.loop|backprop|gradient|optimizer|adam|sgd|learning.rate|loss.function|epoch|batch.size)\b', re.I)),
    ("agent",              re.compile(r'\b(agent|tool.use|react|chain.of.thought|planning|retrieval|rag|function.call)\b', re.I)),
    ("reverse_engineering", re.compile(r'\b(reverse.engineer|disassembl|decompil|ida|ghidra|binary.analysis|shellcode|exploit|payload)\b', re.I)),
    ("operating_system",   re.compile(r'\b(kernel|syscall|interrupt|process|fork|exec|file.system|inode|virtual.memory|page.table)\b', re.I)),
    ("optimization",       re.compile(r'\b(cache|memoiz|big[_-]?o|complexity|profil|benchmark|vectoriz|simd|parallel)\b', re.I)),
]

def classify_topic(text: str) -> str | None:
    """
    Mengembalikan topik pertama yang cocok, atau None jika tidak ada.
    Greedy first-match berdasarkan urutan prioritas di TOPICS.
    """
    for topic_name, pattern in TOPICS:
        if pattern.search(text):
            return topic_name
    return None


def classify_topic_batch(texts: list[str]) -> list[str | None]:
    """Batch version."""
    return [classify_topic(t) for t in texts]


if __name__ == "__main__":
    samples = [
        "We fine-tuned a GPT model using RLHF on a custom dataset.",
        "SELECT * FROM users WHERE id = 1; -- SQL injection",
        "The JWT token contains claims signed with HS256.",
        "Use kubectl apply -f deployment.yaml to create pods.",
        "A simple hello world script in Python.",
        "The malloc call returned NULL, causing a segfault.",
    ]
    for s in samples:
        topic = classify_topic(s)
        print(f"[{topic or 'NONE':>20}] {s[:70]}")
