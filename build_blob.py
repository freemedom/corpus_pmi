"""从五个莱比锡语料的 words.txt 和 inv_w.txt 建 leipzig_blob.sqlite。

不读取、不修改 leipzig.sqlite，也不改 build_db.py、query.py。
句子和多词规则与 build_db.py 里的一份相同，写在本文件里。
postings 每个折叠词一行：data 是按语料、句子、位置排好的差分块。

临时文件在 _build_blob_tmp，日志在 _build_blob.log。
新库若已存在，只删新库再重写。
"""

import array
import gc
import heapq
import sqlite3
import struct
import time
from pathlib import Path

import numpy as np

from postings_codec import Encoder, check_roundtrip, iter_postings
from query_blob import SEP, fold

ROOT = Path(__file__).resolve().parent
DB = ROOT / "leipzig_blob.sqlite"
OLD_NAME = "leipzig.sqlite"
TMP = ROOT / "_build_blob_tmp"
LOG = ROOT / "_build_blob.log"

CORPORA = (
    (0, "fra_news_2020_1M"),
    (1, "fra_news_2022_1M"),
    (2, "fra_news_2023_1M"),
    (3, "fra_news_2024_1M"),
    (4, "fra-ca_web_2020_1M"),
)

# 临时索引一条记录：小端三个 uint32（词形编号、句号、位置），共 12 字节。
REC = struct.Struct("<III")


def log(message):
    print(message, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def words_path(name):
    return ROOT / name / f"{name}-words.txt"


def inv_path(name):
    return ROOT / name / f"{name}-inv_w.txt"


def check_inputs():
    missing = []
    for _, name in CORPORA:
        for path in (words_path(name), inv_path(name)):
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise SystemExit("缺少文件：\n" + "\n".join(missing))


def check_fold():
    assert fold("Été") == fold("ete") == fold("été") == "ete"
    assert fold("Les") == fold("les") == "les"
    assert fold("aujourd'hui") != fold("aujourd’hui")
    assert fold("œuvre") == "oeuvre"
    assert fold("Æ") == "ae"


def collect_folds():
    fold_to_id = {}
    for _, name in CORPORA:
        with words_path(name).open(encoding="utf-8", buffering=1024 * 1024) as handle:
            for line in handle:
                _wid, word, _freq = line.rstrip("\r\n").split("\t", 2)
                key = fold(word)
                if key not in fold_to_id:
                    fold_to_id[key] = len(fold_to_id) + 1
        log(f"词形表已读 {name}，折叠后 {len(fold_to_id)} 个")
    return fold_to_id


def load_words(name, fold_to_id):
    rows = []
    max_id = 0
    with words_path(name).open(encoding="utf-8", buffering=1024 * 1024) as handle:
        for line in handle:
            wid_s, word, _freq = line.rstrip("\r\n").split("\t", 2)
            wid = int(wid_s)
            if wid > max_id:
                max_id = wid
            rows.append((wid, word))
    surfaces = [None] * (max_id + 1)
    form_of = array.array("I", [0]) * (max_id + 1)
    for wid, word in rows:
        surfaces[wid] = word
        form_of[wid] = fold_to_id[fold(word)]
    return surfaces, form_of


def open_db():
    if DB.name == OLD_NAME:
        raise SystemExit("拒绝写入 leipzig.sqlite")
    for extra in ("-journal", "-wal", "-shm"):
        path = Path(str(DB) + extra)
        if path.exists():
            path.unlink()
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(str(DB))
    # 装入期间用 16KB 页、关日志、关同步。写到一半失败就留下坏文件，重跑会删掉重建。
    # 装完后 journal_mode 改回 DELETE、synchronous 改回 NORMAL。
    con.execute("PRAGMA page_size=16384")
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA locking_mode=EXCLUSIVE")
    con.execute("PRAGMA cache_size=-262144")
    con.execute("PRAGMA temp_store=MEMORY")
    con.isolation_level = None
    con.executescript(
        """
        CREATE TABLE corpora (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE forms (
            id INTEGER PRIMARY KEY,
            fold TEXT NOT NULL UNIQUE
        );
        CREATE TABLE sentences (
            corpus INTEGER NOT NULL,
            s_id INTEGER NOT NULL,
            tokens TEXT NOT NULL,
            PRIMARY KEY (corpus, s_id)
        ) WITHOUT ROWID;
        CREATE TABLE postings (
            form_id INTEGER PRIMARY KEY,
            data BLOB NOT NULL
        ) WITHOUT ROWID;
        """
    )
    con.execute("BEGIN")
    con.executemany("INSERT INTO corpora(id, name) VALUES (?, ?)", CORPORA)
    con.execute("COMMIT")
    return con


def insert_forms(con, fold_to_id):
    con.execute("BEGIN")
    items = [(fid, key) for key, fid in fold_to_id.items()]
    items.sort()
    for start in range(0, len(items), 50_000):
        con.executemany(
            "INSERT INTO forms(id, fold) VALUES (?, ?)",
            items[start : start + 50_000],
        )
    con.execute("COMMIT")
    log(f"写入折叠词形 {len(items)}")


def read_inv(name, form_of):
    n_surface = len(form_of)
    fs = array.array("I")
    ss = array.array("I")
    ps = array.array("I")
    ws = array.array("I")
    started = time.perf_counter()
    with inv_path(name).open(encoding="utf-8", buffering=8 * 1024 * 1024) as handle:
        for line_no, line in enumerate(handle, 1):
            wid_s, sid_s, pos_s = line.split("\t", 2)
            wid = int(wid_s)
            if wid >= n_surface or form_of[wid] == 0:
                raise SystemExit(f"{name} 的 inv_w 第 {line_no} 行词号 {wid} 不在 words.txt")
            fs.append(form_of[wid])
            ss.append(int(sid_s))
            ps.append(int(pos_s))
            ws.append(wid)
            if line_no % 5_000_000 == 0:
                log(f"  已读 {line_no} 行，{time.perf_counter() - started:.0f} 秒")
    log(f"  读完 {len(fs)} 行，{time.perf_counter() - started:.0f} 秒")
    # 前四项是缓冲本身。后四项是贴在同一块内存上的 numpy 视图，自己不持有数据。
    return (
        fs,
        ss,
        ps,
        ws,
        np.frombuffer(fs, dtype=np.uint32),
        np.frombuffer(ss, dtype=np.uint32),
        np.frombuffer(ps, dtype=np.uint32),
        np.frombuffer(ws, dtype=np.uint32),
    )


def base_token(wids, surfaces, corpus, sid, pos):
    """同一位置上，不带空格的是句中的词，带空格的是从这里开始的多词条目。

    有的词本身含窄空格（如 11 h），而且该位置只有它一个，这时就保留它。
    """
    plain = []
    spaced = []
    for wid in wids:
        text = surfaces[wid]
        if any(ch.isspace() for ch in text):
            spaced.append(text)
        else:
            plain.append(text)
    if len(plain) == 1:
        return plain[0]
    if len(plain) > 1:
        raise SystemExit(f"语料 {corpus} 句子 {sid} 位置 {pos} 有多个单词：{plain}")
    if len(spaced) == 1:
        return spaced[0]
    raise SystemExit(f"语料 {corpus} 句子 {sid} 位置 {pos} 无法确定词：{spaced}")


def write_sentences(con, corpus, ss, ps, ws, surfaces):
    # 把这个语料的全部出现收成一句一行。位置从 0 起，写进句子后下标仍是它。
    n = int(ss.shape[0])
    log(f"  归并句子 {n} …")
    # 先按句号，同一句再按位置。idx 存的是原数组下标。
    idx = np.lexsort((ps, ss))
    sql = "INSERT INTO sentences(corpus, s_id, tokens) VALUES (?, ?, ?)"
    batch = []
    count = 0
    gaps = 0
    phrases = 0
    con.execute("BEGIN")
    i = 0
    while i < n:
        sid = int(ss[int(idx[i])])
        buf = []
        cur_pos = None
        cur_wids = []

        def flush_pos():
            nonlocal gaps, phrases
            if cur_pos is None:
                return
            if len(cur_wids) > 1:
                phrases += 1
            while len(buf) < cur_pos:
                buf.append("")
                gaps += 1
            buf.append(base_token(cur_wids, surfaces, corpus, sid, cur_pos))

        while i < n:
            j = int(idx[i])
            if int(ss[j]) != sid:
                break
            pos = int(ps[j])
            wid = int(ws[j])
            if cur_pos is None or pos == cur_pos:
                cur_pos = pos
                cur_wids.append(wid)
            else:
                flush_pos()
                cur_pos = pos
                cur_wids = [wid]
            i += 1
        flush_pos()
        batch.append((corpus, sid, SEP.join(buf)))
        if len(batch) >= 2000:
            con.executemany(sql, batch)
            count += len(batch)
            batch.clear()
            if count % 200_000 == 0:
                log(f"  句子已写 {count}")
    if batch:
        con.executemany(sql, batch)
        count += len(batch)
    con.execute("COMMIT")
    del idx
    log(f"  句子 {count}，空位 {gaps}，多词位置 {phrases}")
    return count


def write_bin(path, fs, ss, ps):
    n = int(fs.shape[0])
    log(f"  排序索引 {n} …")
    started = time.perf_counter()
    idx = np.lexsort((ps, ss, fs))
    log(f"  排序完成，{time.perf_counter() - started:.0f} 秒，开始写出")
    dtype = np.dtype([("f", "<u4"), ("s", "<u4"), ("p", "<u4")])
    chunk = 2_000_000
    with path.open("wb", buffering=8 * 1024 * 1024) as handle:
        for start in range(0, n, chunk):
            chosen = idx[start : start + chunk]
            out = np.empty(chosen.shape[0], dtype=dtype)
            out["f"] = fs[chosen]
            out["s"] = ss[chosen]
            out["p"] = ps[chosen]
            out.tofile(handle)
            log(f"  索引已写出 {min(start + chunk, n)}")
    del idx


def build_corpus(con, corpus, name, fold_to_id):
    """处理一个语料，然后把这一份的大数组丢掉。

    句子当场写入 sentences。索引写成已按 (词形, 句号, 位置) 排好的临时文件。
    多词条目仍留在索引里，句子正文只收该位置上的词。
    """
    log(f"开始 {name}")
    started = time.perf_counter()
    surfaces, form_of = load_words(name, fold_to_id)
    arrays = read_inv(name, form_of)
    _fs_arr, _ss_arr, _ps_arr, _ws_arr, fs, ss, ps, ws = arrays
    write_sentences(con, corpus, ss, ps, ws, surfaces)
    bin_path = TMP / f"{corpus}.bin"
    write_bin(bin_path, fs, ss, ps)
    del arrays, fs, ss, ps, ws, surfaces, form_of
    gc.collect()
    log(f"完成 {name}，{time.perf_counter() - started:.0f} 秒，临时文件 {bin_path.stat().st_size / 2**20:.0f} MB")
    return bin_path


def iter_bin(path, corpus):
    size = REC.size
    with path.open("rb", buffering=8 * 1024 * 1024) as handle:
        while True:
            chunk = handle.read(size * 200_000)
            if not chunk:
                break
            if len(chunk) % size != 0:
                raise SystemExit(f"索引文件不完整：{path}")
            for offset in range(0, len(chunk), size):
                form_id, s_id, pos = REC.unpack_from(chunk, offset)
                yield (form_id, corpus, s_id, pos)


def merge_postings(con, bins):
    """五个临时文件按 (词形, 语料, 句号, 位置) 归并。完全相同的只留第一条。

    同一个词形的出现收成一块差分数据，词形一变就写入上一块。
    """
    log("合并索引写入差分块…")
    started = time.perf_counter()
    streams = [iter_bin(path, corpus) for corpus, path in bins]
    sql = "INSERT INTO postings(form_id, data) VALUES (?, ?)"
    batch = []
    hits = 0
    forms = 0
    dups = 0
    payload = 0
    prev = None
    cur_form = None
    enc = Encoder()
    con.execute("BEGIN")

    def flush_form():
        nonlocal forms, payload
        if cur_form is None:
            return
        data = enc.finish()
        payload += len(data)
        batch.append((cur_form, data))
        forms += 1
        if len(batch) >= 2000:
            con.executemany(sql, batch)
            batch.clear()
            if forms % 100_000 == 0:
                con.execute("COMMIT")
                con.execute("BEGIN")
                log(f"  词形 {forms}，出现 {hits}，{time.perf_counter() - started:.0f} 秒")

    for rec in heapq.merge(*streams):
        if rec == prev:
            dups += 1
            continue
        prev = rec
        form_id, corpus, sid, pos = rec
        if form_id != cur_form:
            flush_form()
            cur_form = form_id
        enc.add(corpus, sid, pos)
        hits += 1
        if hits % 2_000_000 == 0:
            log(f"  已归并 {hits}，{time.perf_counter() - started:.0f} 秒")
    flush_form()
    if batch:
        con.executemany(sql, batch)
    con.execute("COMMIT")
    log(
        f"索引写入 {forms} 个词形、{hits} 次出现，"
        f"重复跳过 {dups}，正文 {payload / 2**20:.0f} MB，"
        f"{time.perf_counter() - started:.0f} 秒"
    )
    return hits


def self_check(con):
    form_id = con.execute("SELECT id FROM forms WHERE fold = ?", (fold("les"),)).fetchone()[0]
    blob = con.execute(
        "SELECT data FROM postings WHERE form_id = ?",
        (form_id,),
    ).fetchone()
    if blob is None:
        raise SystemExit("自检失败：les 没有差分块")
    corpus, s_id, pos = next(iter_postings(blob[0], 1))
    tokens = con.execute(
        "SELECT tokens FROM sentences WHERE corpus = ? AND s_id = ?",
        (corpus, s_id),
    ).fetchone()[0].split(SEP)
    if fold(tokens[pos]) != "les":
        raise SystemExit(f"自检失败：首条命中是 {tokens[pos]!r}")
    known = con.execute(
        "SELECT tokens FROM sentences WHERE corpus = 0 AND s_id = 875686"
    ).fetchone()
    if known is None or "les" not in known[0].split(SEP):
        raise SystemExit("自检失败：2020 句子 875686 里没有 les")
    phrase = con.execute(
        "SELECT tokens FROM sentences WHERE corpus = 0 AND s_id = 545765"
    ).fetchone()
    phrase_tokens = phrase[0].split(SEP) if phrase else []
    if len(phrase_tokens) < 9 or phrase_tokens[0] != "Le" or phrase_tokens[8] != "Xbox":
        raise SystemExit("自检失败：多词条目盖住了句中的词 " + " ".join(phrase_tokens))
    n = con.execute("SELECT COUNT(*) FROM sentences WHERE corpus = 0").fetchone()[0]
    plan = con.execute(
        "EXPLAIN QUERY PLAN SELECT data FROM postings WHERE form_id = ?",
        (form_id,),
    ).fetchall()
    log(f"自检通过。2020 句子数 {n}。首条命中位置 {pos}：{tokens[pos]}")
    log("查询计划：" + " | ".join(str(row) for row in plan))


def compare_with_old():
    """只读打开旧库，比 les、de、Xbox 的前 100 条窗口。建库过程不走这里。"""
    import query
    import query_blob

    old_path = ROOT / OLD_NAME
    if not old_path.exists():
        raise SystemExit(f"对照需要旧库：{old_path}")
    old = query.connect()
    new = query_blob.connect()
    old_names = dict(old.execute("SELECT id, name FROM corpora"))
    new_names = dict(new.execute("SELECT id, name FROM corpora"))
    for word in ("les", "de", "Xbox"):
        key_old, hits_old = query.search(old, word, 100, old_names)
        key_new, hits_new = query_blob.search(new, word, 100, new_names)
        if key_old != key_new or hits_old != hits_new:
            n = 0
            for left, right in zip(hits_old, hits_new):
                if left != right:
                    break
                n += 1
            raise SystemExit(
                f"对照失败：{word} 折叠 {key_old} / {key_new}，"
                f"旧库 {len(hits_old)} 条，新库 {len(hits_new)} 条，"
                f"前 {n} 条相同"
            )
        log(f"对照通过：{word} 前 {len(hits_old)} 条")
    old.close()
    new.close()


def reset_tmp():
    if TMP.exists():
        for child in TMP.iterdir():
            child.unlink()
        TMP.rmdir()
    TMP.mkdir()


def main():
    started = time.perf_counter()
    LOG.write_text("", encoding="utf-8")
    check_roundtrip()
    check_fold()
    check_inputs()
    reset_tmp()
    con = open_db()
    fold_to_id = collect_folds()
    insert_forms(con, fold_to_id)
    bins = []
    for corpus, name in CORPORA:
        bins.append((corpus, build_corpus(con, corpus, name, fold_to_id)))
    merge_postings(con, bins)
    con.execute("PRAGMA journal_mode=DELETE")
    con.execute("PRAGMA synchronous=NORMAL")
    self_check(con)
    con.close()
    compare_with_old()
    for _, path in bins:
        path.unlink(missing_ok=True)
    TMP.rmdir()
    log(f"完成。库 {DB.stat().st_size / 2**20:.0f} MB，总用时 {time.perf_counter() - started:.0f} 秒")


if __name__ == "__main__":
    main()
