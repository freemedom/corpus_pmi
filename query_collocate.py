"""查询词的窗口写入 txt，搭配统计另写入一个文件。

命中行是前后各 10 个词，查询词不加 << >>，存 collocate/{折叠词}.txt。
搭配只在这次取出的命中上计算，左右各 5 个词，词先折叠再计数，存 collocate/{折叠词}_pmi.txt。
标点不进表。门槛与 countLeftRightPmi.py 相同。
"""

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path

from postings_codec import iter_postings
from query_blob import RADIUS, SEP, connect, fold

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "collocate"

SPAN = 5
PMI_LIMIT = 2

OFFSET = 5
FREQ_WORD = 5 + OFFSET * 2
FREQ_BIGRAM = 5 + OFFSET
FREQ_PAIR = 5 + OFFSET
WORD_WIDTH = 10
BIGRAM_WIDTH = 15


def format_window(tokens, pos):
    """取出查询词前后各 RADIUS 个词，连同查询词本身。

    pos 是查询词在这句里的下标。句子不够长就停在句首或句尾。
    切片右端要 +1，因为右边界本身不算进去，这样查询词后面正好留 RADIUS 个词。
    词与词用空格接上，查询词外面不加 << >>。
    """
    lo = max(0, pos - RADIUS)
    hi = min(len(tokens), pos + RADIUS + 1)
    return " ".join(tokens[lo:hi])


def is_punct(key):
    """折叠后没有字母也没有数字，当作标点。"""
    return not any(ch.isalpha() or ch.isdigit() for ch in key)


def pmi(ab_freq, a_freq, b_freq, corpus_size):
    try:
        return math.log((ab_freq * corpus_size) / (a_freq * b_freq * SPAN)) / math.log(2)
    except (ValueError, ZeroDivisionError):
        return float("-inf")


def gather(con, word, limit, names):
    key = fold(word)
    row = con.execute("SELECT id FROM forms WHERE fold = ?", (key,)).fetchone()
    if row is None:
        return key, []
    blob = con.execute(
        "SELECT data FROM postings WHERE form_id = ?",
        (row[0],),
    ).fetchone()
    if blob is None:
        return key, []
    hits = []
    sent_sql = "SELECT tokens FROM sentences WHERE corpus = ? AND s_id = ?"
    con.execute("BEGIN")
    try:
        for corpus, s_id, pos in iter_postings(blob[0], limit):
            sent = con.execute(sent_sql, (corpus, s_id)).fetchone()
            if sent is None:
                raise SystemExit(f"句子不在库里：语料 {corpus} 句号 {s_id}")
            tokens = sent[0].split(SEP)
            folded = [fold(tok) for tok in tokens]
            hits.append((names[corpus], format_window(tokens, pos), folded, pos))
    finally:
        con.execute("ROLLBACK")
    return key, hits


def tally(hits, pmi_limit):
    """tally：把命中汇总成次数，供后面算 PMI。

    每条命中算一行。同一句出现两次就计两次。
    """
    target_freq = len(hits)
    corpus_size = 0
    word_freq = Counter()
    bigram_freq = Counter()
    left_words = Counter()
    right_words = Counter()
    left_bigrams = Counter()
    right_bigrams = Counter()
    pairs = Counter()

    for _name, _line, folded, pos in hits:
        corpus_size += len(folded)  # 这一句有多少个词
        word_freq.update(folded)
        for i in range(len(folded) - 1):
            bigram_freq[f"{folded[i]} {folded[i + 1]}"] += 1

        left = folded[max(0, pos - SPAN) : pos]
        right = folded[pos + 1 : pos + 1 + SPAN]
        left_words.update(left)
        right_words.update(right)
        for i in range(len(left) - 1):
            if is_punct(left[i]) or is_punct(left[i + 1]):
                continue
            left_bigrams[f"{left[i]} {left[i + 1]}"] += 1
        for i in range(len(right) - 1):
            if is_punct(right[i]) or is_punct(right[i + 1]):
                continue
            right_bigrams[f"{right[i]} {right[i + 1]}"] += 1
        for lw in left:
            if is_punct(lw):
                continue
            for rw in right:
                if is_punct(rw):
                    continue
                pairs[f"{lw}@@@{rw}"] += 1

    def word_rows(counts):
        rows = []
        for word, co_freq in counts.items():
            if is_punct(word):
                continue
            score = pmi(co_freq, target_freq, word_freq[word], corpus_size)
            if score > pmi_limit and co_freq >= FREQ_WORD:
                rows.append((word, score, co_freq))
        rows.sort(key=lambda item: item[2], reverse=True)
        return rows

    def bigram_rows(counts):
        rows = []
        for gram, co_freq in counts.items():
            score = pmi(co_freq, target_freq, bigram_freq[gram], corpus_size)
            if score > pmi_limit and co_freq >= FREQ_BIGRAM:
                rows.append((gram, score, co_freq))
        rows.sort(key=lambda item: item[2], reverse=True)
        return rows

    pair_rows = []
    for pair, co_freq in pairs.items():
        left_word, right_word = pair.split("@@@")
        score = pmi(
            co_freq,
            1,
            word_freq[left_word] * word_freq[right_word],
            corpus_size,
        )
        if score > pmi_limit and co_freq >= FREQ_PAIR:
            pair_rows.append((f"{left_word}-{right_word}", score, co_freq))
    pair_rows.sort(key=lambda item: item[2], reverse=True)

    return {
        "pmi_limit": pmi_limit,
        "target_freq": target_freq,
        # 这些命中的词数之和，同一句出现两次就加两次。PMI 用它当语料规模，不是五个语料的总词数。
        "corpus_size": corpus_size,
        "left_words": word_rows(left_words),
        "right_words": word_rows(right_words),
        "left_bigrams": bigram_rows(left_bigrams),
        "right_bigrams": bigram_rows(right_bigrams),
        "pairs": pair_rows,
    }


def pad(text, width):
    return " " * max(0, width - len(text))


def write_hits(path, hits):
    lines = [f"[{name}] {line}" for name, line, _folded, _pos in hits]
    text = "\n".join(lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def write_pmi(path, stats):
    lines = [
        "出现频率",
        f"取出的命中 {stats['target_freq']} 次",
        f"这些命中的总词数: {stats['corpus_size']}",
        "",
        f"左侧最强搭配（按频率排序）: {FREQ_WORD} {stats['pmi_limit']:g}",
        "左单词\t\tPMI值\t共现频率",
    ]
    for word, score, co_freq in stats["left_words"]:
        lines.append(f"左 {word}{pad(word, WORD_WIDTH)}{score:.2f}\t{co_freq}")
    lines.append("")
    lines.append(f"左侧最常见的两词组合及其频率: {FREQ_BIGRAM}")
    lines.append("词组\t\tPMI值\t频率")
    for gram, score, co_freq in stats["left_bigrams"]:
        lines.append(f"左 {gram}{pad(gram, BIGRAM_WIDTH)}{score:.2f}\t{co_freq}")
    lines.append("")
    lines.append("右侧最强搭配（按频率排序）:")
    lines.append("右单词\t\tPMI值\t共现频率")
    for word, score, co_freq in stats["right_words"]:
        lines.append(f"右 {word}{pad(word, WORD_WIDTH)}{score:.2f}\t{co_freq}")
    lines.append("")
    lines.append("右侧最常见的两词组合及其频率:")
    lines.append("词组\t\tPMI值\t频率")
    for gram, score, co_freq in stats["right_bigrams"]:
        lines.append(f"右 {gram}{pad(gram, BIGRAM_WIDTH)}{score:.2f}\t{co_freq}")
    lines.append("")
    lines.append(f"左右词对的搭配（按频率排序）: {FREQ_PAIR}")
    lines.append("左词-右词" + pad("左词-右词", BIGRAM_WIDTH) + "\tPMI值\t共现频率")
    for pair, score, co_freq in stats["pairs"]:
        lines.append(f"{pair}{pad(pair, BIGRAM_WIDTH)}{score:.2f}\t{co_freq}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="把查询窗口写入 collocate/{折叠词}.txt，搭配统计写入 collocate/{折叠词}_pmi.txt。"
    )
    parser.add_argument("words", nargs="+", help="1 到 5 个词")
    parser.add_argument(
        "--limit",
        type=int,
        default=50000,
        help="每个词最多取多少条，默认 50000",
    )
    parser.add_argument(
        "--pmi-limit",
        type=float,
        default=PMI_LIMIT,
        help="PMI 下限，词、词组、左右词对共用，保留大于这个值的行，默认 2",
    )
    args = parser.parse_args(argv)
    if not 1 <= len(args.words) <= 5:
        parser.error("请输入 1 到 5 个词")
    if args.limit < 1:
        parser.error("--limit 至少为 1")

    OUT_DIR.mkdir(exist_ok=True)
    con = connect()
    names = dict(con.execute("SELECT id, name FROM corpora"))
    for word in args.words:
        started = time.perf_counter()
        key, hits = gather(con, word, args.limit, names)
        hits_path = OUT_DIR / f"{key}.txt"
        pmi_path = OUT_DIR / f"{key}_pmi.txt"
        write_hits(hits_path, hits)
        write_pmi(pmi_path, tally(hits, args.pmi_limit))
        elapsed = time.perf_counter() - started
        print(
            f"（{word} 折叠为 {key}，返回 {len(hits)} 条，上限 {args.limit}，"
            f"PMI 下限 {args.pmi_limit:g}，写入 {hits_path.name} 和 {pmi_path.name}，"
            f"{elapsed:.2f} 秒）",
            file=sys.stderr,
        )
    con.close()


if __name__ == "__main__":
    main()
