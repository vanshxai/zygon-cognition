"""Plain text out of an uploaded file: pdf, docx, pptx, odt, epub, html, or any text-like file."""
import html as htmllib, io, os, re, zipfile
import pypdf


def html_to_text(h):
    h = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header|aside|form)[^>]*>.*?</\1>", " ", h)
    paras = re.findall(r"(?is)<(?:p|li|h[1-4])[^>]*>(.*?)</(?:p|li|h[1-4])>", h)
    txt = "\n\n".join(htmllib.unescape(re.sub(r"<[^>]+>", " ", p)) for p in paras) if paras else htmllib.unescape(re.sub(r"<[^>]+>", " ", h))
    return re.sub(r"[ \t]+", " ", txt)


def extract_upload(name, raw):
    """returns (text, pages or None)"""
    ext = os.path.splitext(name.lower())[1]
    if ext == ".pdf":
        r = pypdf.PdfReader(io.BytesIO(raw)); return "\n\n".join((pg.extract_text() or "") for pg in r.pages), len(r.pages)
    if ext in (".docx", ".pptx", ".epub", ".odt"):
        z = zipfile.ZipFile(io.BytesIO(raw)); out = []
        if ext == ".docx": parts = ["word/document.xml"]
        elif ext == ".pptx": parts = sorted((n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml", n)), key=lambda n: int(re.findall(r"\d+", n)[0]))
        elif ext == ".odt": parts = ["content.xml"]
        else: parts = [n for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]
        for n in parts:
            x = z.read(n).decode("utf-8", "replace")
            if ext == ".epub": out.append(html_to_text(x)); continue
            x = re.sub(r"</(?:w:p|a:p|text:p|text:h)>", "\n\n", x); out.append(htmllib.unescape(re.sub(r"<[^>]+>", "", x)))
        return "\n\n".join(out), None
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"): t = raw.decode("utf-16", "replace")
    else: t = raw.decode("utf-8-sig", "replace")
    if ext in (".html", ".htm"): t = html_to_text(t)
    return t, None
