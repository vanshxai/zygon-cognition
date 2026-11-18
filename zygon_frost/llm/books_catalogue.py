"""Find books around a topic, 50 to 100 pages each, in Project Gutenberg's official catalogue (public-domain books).
    python books_catalogue.py [topic words...]          -> model/books/catalogue_<topic>.json and .html
Pages are estimated as (text size - 18,000 bytes of licence text) / 5.7 bytes per word / 300 words per page.
Sizes come from a header request (curl, which validates certificates) for each candidate; nothing else is downloaded.
"""
import concurrent.futures as cf, csv, html, json, os, re, subprocess, sys, time

sys.stdout.reconfigure(encoding="utf-8"); csv.field_size_limit(10 ** 7)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
BOOKS = os.path.join(ROOT, "model", "books")
TOPICS = {"science": ["science", "physics", "chemistry", "astronomy", "biology", "geology", "natural history", "mathematics", "electricity", "botany", "zoology", "evolution", "meteorology", "anatomy", "physiology", "geometry", "algebra", "astronomy", "mechanics", "optics", "heat", "light", "magnetism", "scientific"]}
LO, HI = 50, 100


def pages(size): return (size - 18000) / 5.7 / 300


def head(bid):
    url = f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt"
    try:
        out = subprocess.run(["curl", "-sS", "-m", "25", "-I", "-A", "FrostLang/1.0 (personal research)", url], capture_output=True, timeout=40).stdout.decode("latin-1")
        if " 200" not in out.split("\n")[0]: return bid, None
        m = re.search(r"content-length:\s*(\d+)", out, re.I); return bid, int(m.group(1)) if m else None
    except Exception: return bid, None


def main():
    topic = " ".join(sys.argv[1:]).lower() or "science"; kws = TOPICS.get(topic, [topic]); cat = os.path.join(BOOKS, "pg_catalog.csv"); rows = []
    for r in csv.DictReader(open(cat, encoding="utf-8", newline="")):
        if r["Type"] != "Text" or r["Language"].strip() != "en": continue
        text = (r["Subjects"] + " " + r["Bookshelves"]).lower(); hit = sum(1 for k in set(kws) if re.search(r"\b" + re.escape(k) + r"\b", text))
        if hit: rows.append({"id": int(r["Text#"]), "title": " ".join(r["Title"].split()), "authors": r["Authors"], "subjects": r["Subjects"][:160], "issued": r["Issued"], "hits": hit})
    rows.sort(key=lambda x: (-x["hits"], x["id"])); print(f"{len(rows)} English texts about '{topic}' in the catalogue; checking their sizes...", flush=True)
    found, t0, done = [], time.time(), 0
    with cf.ThreadPoolExecutor(14) as ex:
        for chunk_start in range(0, len(rows), 140):
            chunk = rows[chunk_start:chunk_start + 140]
            for bid, size in ex.map(head, [r["id"] for r in chunk]):
                done += 1; r = next(x for x in chunk if x["id"] == bid)
                if size and LO <= pages(size) <= HI: r["bytes"] = size; r["pages"] = round(pages(size), 1); r["words"] = int((size - 18000) / 5.7); found.append(r)
            print(f"  checked {done}/{len(rows)}; {len(found)} books of {LO}-{HI} pages so far ({time.time() - t0:.0f}s)", flush=True)
            if len(found) >= 500: break
    found = found[:500]
    out = {"topic": topic, "pages_rule": f"{LO} to {HI} pages, page = 300 words", "count": len(found), "checked": done, "seconds": round(time.time() - t0), "books": found}
    json.dump(out, open(os.path.join(BOOKS, f"catalogue_{topic}.json"), "w", encoding="utf-8"), indent=1)
    rows_html = "".join(f"<tr><td>{i + 1}</td><td>{b['id']}</td><td>{html.escape(b['title'][:90])}</td><td>{html.escape(b['authors'][:40])}</td><td>{b['pages']}</td><td>{b['words']:,}</td></tr>" for i, b in enumerate(found))
    open(os.path.join(BOOKS, f"catalogue_{topic}.html"), "w", encoding="utf-8").write(f"<!doctype html><meta charset=utf-8><title>{len(found)} books about {topic}</title><style>body{{font:13px system-ui;margin:20px}}td,th{{border-bottom:1px solid #ddd;padding:3px 8px;text-align:left}}</style><h2>{len(found)} public-domain books about {topic}, {LO} to {HI} pages</h2><p>Source: Project Gutenberg catalogue. Pages = (text size - 18 KB licence) / 5.7 bytes per word / 300 words per page.</p><table><tr><th>#</th><th>Gutenberg id</th><th>Title</th><th>Author</th><th>Pages</th><th>Words</th></tr>{rows_html}</table>")
    print(f"DONE: {len(found)} books in {out['seconds']} s (checked {done})", flush=True)


if __name__ == "__main__":
    main()
