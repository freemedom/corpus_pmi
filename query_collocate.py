"""查询词的窗口写入 txt，搭配统计另写入一个文件。

命中行是前后各 10 个词，查询词不加 << >>，存 collocate/{折叠词}.txt。
一次传入的多个词视为同一个节点，折叠形式用 + 接成文件名。
搭配只在这次取出的命中上计算，左右各 5 个词，词先折叠再计数，存 collocate/{折叠词}_pmi.txt。
标点不进表。门槛与 countLeftRightPmi.py 相同。
"""

import argparse
import heapq
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


def distinct_folds(words):
    """按输入顺序折叠。折成同一个形式的只留第一次。"""
    keys = []
    seen = set()
    for word in words:
        key = fold(word)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def postings_for(con, key, limit):
    """取出一个折叠形式最靠前的出现位置，不读句子文本。

    key 已经折叠过。forms 表用折叠形式查到词条编号；没有这个形式就返回空列表。
    出现记录不在这张表里，而在 postings 里按词条编号存成一块字节。
    那一块按语料、句子、位置编码，iter_postings 按这个顺序解开，数到 limit 条就停。
    返回的是 (语料号, 句子号, 句中位置)，后面再按这些编号去读句子。
    """
    row = con.execute("SELECT id FROM forms WHERE fold = ?", (key,)).fetchone()
    if row is None:
        return []
    blob = con.execute(
        "SELECT data FROM postings WHERE form_id = ?",
        (row[0],),
    ).fetchone()
    if blob is None:
        return []
    return list(iter_postings(blob[0], limit))


def gather(con, words, limit, names):
    """多个词视为一个节点，返回 (文件名主干, 命中列表)。

    先把输入折成形式，重复的去掉，再用 + 接成文件名主干。
    每个形式的出现记录已经按语料、句子、位置排好。各取自己最靠前的 limit 条。
    库里没有的形式就是一条空流，文件名里仍保留它。

    这些流再按同一顺序混成一条。混完只留前 limit 条，所以节点一共只有这么多命中。
    某个形式若排在自己的第 limit 条之后，它前面至少已经有 limit 条自己的出现，
    合并后也到不了前 limit 名，一开始不取它是安全的。

    位置名单定下来之后才去读句子，避免把将被丢掉的命中也读出来。
    一次只读事务里把这些句子取完，读完就回滚。
    每条命中是：语料名、前后各 10 词的窗口、整句折叠后的词、查询词在句中的下标。
    """
    keys = distinct_folds(words)
    stem = "+".join(keys)
    streams = [postings_for(con, key, limit) for key in keys]
    streams = [stream for stream in streams if stream]
    chosen = []
    # 各流内部已经有序，merge 按语料、句子、位置往外出，不必先合成一张大表再排序。
    for corpus, s_id, pos in heapq.merge(*streams):
        chosen.append((corpus, s_id, pos))
        if len(chosen) >= limit:
            break
    hits = []
    sent_sql = "SELECT tokens FROM sentences WHERE corpus = ? AND s_id = ?"
    con.execute("BEGIN")
    try:
        for corpus, s_id, pos in chosen:
            sent = con.execute(sent_sql, (corpus, s_id)).fetchone()
            if sent is None:
                raise SystemExit(f"句子不在库里：语料 {corpus} 句号 {s_id}")
            tokens = sent[0].split(SEP)
            folded = [fold(tok) for tok in tokens]
            hits.append((names[corpus], format_window(tokens, pos), folded, pos))
    finally:
        con.execute("ROLLBACK")
    return stem, hits


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
        description="把查询窗口写入 collocate/{折叠词}.txt，搭配统计写入 collocate/{折叠词}_pmi.txt。多个词视为同一个节点。"
    )
    parser.add_argument("words", nargs="+", help="1 到 5 个词，多个词视为同一个节点")
    parser.add_argument(
        "--limit",
        type=int,
        default=50000,
        help="合并后的节点最多取多少条，默认 50000",
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
    started = time.perf_counter()
    stem, hits = gather(con, args.words, args.limit, names)
    hits_path = OUT_DIR / f"{stem}.txt"
    pmi_path = OUT_DIR / f"{stem}_pmi.txt"
    write_hits(hits_path, hits)
    write_pmi(pmi_path, tally(hits, args.pmi_limit))
    elapsed = time.perf_counter() - started
    label = "、".join(args.words)
    print(
        f"（{label} 折叠为 {stem}，返回 {len(hits)} 条，上限 {args.limit}，"
        f"PMI 下限 {args.pmi_limit:g}，写入 {hits_path.name} 和 {pmi_path.name}，"
        f"{elapsed:.2f} 秒）",
        file=sys.stderr,
    )
    con.close()


if __name__ == "__main__":
    main()
