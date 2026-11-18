"""Look something up on several sources, one by one, and return organised results.

    python web_search.py "speed of light"        (prints every source in turn)

Sources: Wikipedia, YouTube, DuckDuckGo, Wiktionary, Open Library, Stack Overflow, Hacker News, arXiv (public APIs). Google was removed: it blocks automated searches.
"""
import gzip, html, json, os, re, subprocess, sys, time
import urllib.parse, urllib.request
import xml.etree.ElementTree as ET

UA = "ZYGON/1.0 (personal research program running on the user's own PC)"


def _get(url, timeout=12, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
        if data[:2] == b"\x1f\x8b": data = gzip.decompress(data)
    return data.decode("utf-8", "replace")


def _text(h):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", h or ""))).strip()


def res(source, ok, summary="", items=None, text="", note=""):
    return {"source": source, "ok": ok, "summary": summary, "items": items or [], "text": text, "note": note, "link": ""}


# ---------------------------------------------------------------- the others
def wikipedia(query):
    s = json.loads(_get("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=3&srsearch=" + urllib.parse.quote(query)))
    hits = s.get("query", {}).get("search", [])
    if not hits: return res("Wikipedia", False, note="nothing found")
    top = hits[0]["title"]; d = json.loads(_get("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(top.replace(" ", "_"))))
    ex = d.get("extract", ""); items = [{"title": h["title"], "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(h["title"].replace(" ", "_")), "detail": _text(h.get("snippet", ""))} for h in hits]
    return res("Wikipedia", True, summary=ex, items=items, text=ex, note=top)


def duckduckgo(query):
    d = json.loads(_get("https://api.duckduckgo.com/?format=json&no_html=1&skip_disambig=1&q=" + urllib.parse.quote(query)))
    ab = d.get("AbstractText") or d.get("Answer") or ""; items = []
    for t in d.get("RelatedTopics", [])[:4]:
        if isinstance(t, dict) and t.get("Text"): items.append({"title": t["Text"][:110], "url": t.get("FirstURL", ""), "detail": ""})
    if not ab and not items: return res("DuckDuckGo", False, note="no instant answer")
    return res("DuckDuckGo", True, summary=ab or "Related topics:", items=items, text=ab)


def youtube(query):
    h = _get("https://www.youtube.com/results?hl=en&search_query=" + urllib.parse.quote(query), timeout=15)
    m = re.search(r"var ytInitialData = (\{.*?\});</script>", h, re.S)
    if not m: return res("YouTube", False, note="YouTube's page had no results I could read")
    data = json.loads(m.group(1)); items = []

    def walk(o):
        if len(items) >= 5: return
        if isinstance(o, dict):
            if "videoRenderer" in o:
                v = o["videoRenderer"]; t = "".join(r.get("text", "") for r in v.get("title", {}).get("runs", [])); ch = "".join(r.get("text", "") for r in v.get("ownerText", {}).get("runs", []))
                det = ", ".join(x for x in (ch, v.get("lengthText", {}).get("simpleText", ""), v.get("viewCountText", {}).get("simpleText", ""), v.get("publishedTimeText", {}).get("simpleText", "")) if x)
                items.append({"title": t, "url": "https://www.youtube.com/watch?v=" + v.get("videoId", ""), "detail": det}); return
            for x in o.values(): walk(x)
        elif isinstance(o, list):
            for x in o: walk(x)
    walk(data)
    if not items: return res("YouTube", False, note="no videos found")
    return res("YouTube", True, summary="Top videos:", items=items, text="Videos about this: " + "; ".join(i["title"] for i in items[:3]) + ".")


def wiktionary(query):
    if len(query.split()) > 2: return res("Wiktionary", False, note="only for single words")
    d = json.loads(_get("https://en.wiktionary.org/api/rest_v1/page/definition/" + urllib.parse.quote(query.strip().lower())))
    ens = d.get("en", [])
    if not ens: return res("Wiktionary", False, note="no definition")
    lines = []
    for e in ens[:3]:
        defs = [_text(x.get("definition", "")) for x in e.get("definitions", [])[:2] if x.get("definition")]
        if defs: lines.append(f"{e.get('partOfSpeech', '')}: " + "; ".join(defs))
    return res("Wiktionary", True, summary=" | ".join(lines), items=[], text=f"{query} ({lines[0]})" if lines else "")


def openlibrary(query):
    d = json.loads(_get("https://openlibrary.org/search.json?limit=4&q=" + urllib.parse.quote(query)))
    items = [{"title": x.get("title", ""), "url": "https://openlibrary.org" + x.get("key", ""), "detail": ", ".join(x.get("author_name", [])[:2]) + (f" ({x['first_publish_year']})" if x.get("first_publish_year") else "")} for x in d.get("docs", [])[:4]]
    if not items: return res("Open Library", False, note="no books found")
    return res("Open Library", True, summary="Books:", items=items, text="Books about this: " + "; ".join(f"{i['title']} by {i['detail']}" for i in items[:3]) + ".")


def stackoverflow(query):
    d = json.loads(_get("https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&pagesize=4&site=stackoverflow&q=" + urllib.parse.quote(query)))
    items = [{"title": html.unescape(x["title"]), "url": x["link"], "detail": f"{x.get('score', 0)} votes, {'answered' if x.get('is_answered') else 'not answered'}"} for x in d.get("items", [])[:4]]
    if not items: return res("Stack Overflow", False, note="no programming questions found")
    return res("Stack Overflow", True, summary="Programming questions:", items=items, text="Programming questions about this: " + "; ".join(i["title"] for i in items[:3]) + ".")


def hackernews(query):
    d = json.loads(_get("https://hn.algolia.com/api/v1/search?tags=story&hitsPerPage=4&query=" + urllib.parse.quote(query)))
    items = [{"title": x.get("title", ""), "url": x.get("url") or f"https://news.ycombinator.com/item?id={x.get('objectID')}", "detail": f"{x.get('points', 0)} points, {x.get('num_comments', 0)} comments"} for x in d.get("hits", [])[:4] if x.get("title")]
    if not items: return res("Hacker News", False, note="no stories found")
    return res("Hacker News", True, summary="Discussions:", items=items, text="")


def arxiv(query):
    x = _get("https://export.arxiv.org/api/query?max_results=3&search_query=all:" + urllib.parse.quote(query))
    ns = {"a": "http://www.w3.org/2005/Atom"}; root = ET.fromstring(x); items, first = [], ""
    for e in root.findall("a:entry", ns)[:3]:
        t = re.sub(r"\s+", " ", e.findtext("a:title", "", ns)).strip(); sm = re.sub(r"\s+", " ", e.findtext("a:summary", "", ns)).strip(); u = e.findtext("a:id", "", ns)
        items.append({"title": t, "url": u, "detail": sm[:160] + "..."}); first = first or f"{t}: {sm[:400]}"
    if not items: return res("arXiv", False, note="no research papers found")
    return res("arXiv", True, summary="Research papers:", items=items, text=first)


PROVIDERS = [("Wikipedia", wikipedia), ("YouTube", youtube), ("DuckDuckGo", duckduckgo), ("Wiktionary", wiktionary), ("Open Library", openlibrary), ("Stack Overflow", stackoverflow), ("Hacker News", hackernews), ("arXiv", arxiv)]


def search_all(query, enabled=None, on_result=None):
    """ask each source in turn; on_result(result) is called as soon as a source has answered"""
    out = []
    for name, fn in PROVIDERS:
        if enabled is not None and name not in enabled: continue
        t = time.time()
        try: r = fn(query)
        except Exception as e:
            why = "its security certificate could not be verified from this PC, so I skipped it" if "CERTIFICATE_VERIFY_FAILED" in repr(e) else f"could not reach it ({type(e).__name__})"
            r = res(name, False, note=why)
        r["seconds"] = round(time.time() - t, 1); out.append(r)
        if on_result: on_result(r)
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    q = " ".join(sys.argv[1:]) or "speed of light"
    for r in search_all(q):
        print(f"\n== {r['source']}  ({r['seconds']} s)  {'ok' if r['ok'] else 'no: ' + r['note']}")
        if r["ok"]:
            print("  " + r["summary"][:400])
            for i in r["items"][:4]: print(f"   - {i['title'][:90]}  {i['detail'][:60]}  {i['url'][:70]}")
