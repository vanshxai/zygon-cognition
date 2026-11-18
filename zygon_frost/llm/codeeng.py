"""Frost Code: Frost Lang plus coding lessons (Python, C) and tools that build small web apps by calling functions."""
import re, time
import json, os, subprocess, sys, urllib.request
import lang, code_data, webapps, fullstack


class Code(lang.Lang):
    def teach_code(self):
        t0 = time.perf_counter(); rows = code_data.lesson_rows(); words = 0
        for cat, wordings, reply in rows:
            self.learn_lesson(cat, wordings, [reply]); words += len(reply.split())
        return {"lessons": len(rows), "wordings": sum(len(w) for _, w, _ in rows), "words": words, "seconds": time.perf_counter() - t0}

    def lesson(self, q, prefix=None):
        low = q.lower()
        prefix = prefix or ("C " if re.search(r"(?<![a-z+#])c(?![a-z+#])", low) and "python" not in low else "Python " if "python" in low else None)      # which language was asked for
        return super().lesson(q, prefix)

    def fullstack(self, q, t0):
        """description -> a full-stack project folder + zip, started and checked like a browser would"""
        spec = fullstack.parse_spec(q); info = fullstack.build_zip(spec)
        if not info["ok"]: return {"kind": "told", "text": "I could not make that app: " + info.get("why", "")}
        port = webapps.free_port(); env = dict(os.environ, APP_PASSWORD="")
        proc = subprocess.Popen([sys.executable, os.path.join(info["dir"], "server.py"), str(port)], cwd=info["dir"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        url = f"http://127.0.0.1:{port}/"; up = False
        for _ in range(40):
            try:
                if urllib.request.urlopen(url, timeout=2).status == 200: up = True; break
            except Exception: time.sleep(0.2)
        webapps.RUNNING[info["name"]] = {"proc": proc, "url": url, "dir": info["dir"]}
        key = spec["fields"][0]["key"]; checks = {"page": up}
        def call(path, method="GET", body=None):
            r = urllib.request.urlopen(urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode() if body is not None else None, method=method), timeout=5); return r.status, r.read().decode()
        try:
            c, b = call("/api/items", "POST", {key: "test"}); checks["add"] = c == 201; i = json.loads(b)["id"]; checks["search"] = "test" in call("/api/items?q=test")[1]
            checks["edit"] = call(f"/api/items/{i}", "PUT", {key: "test2"})[0] == 200; checks["stats"] = call("/api/stats")[0] == 200; checks["csv"] = call("/api/export.csv")[0] == 200; checks["delete"] = call(f"/api/items/{i}", "DELETE")[0] == 200
        except Exception as e: checks["error"] = repr(e)
        ut = subprocess.run([sys.executable, "-m", "unittest", "discover", "tests"], cwd=info["dir"], capture_output=True, text=True, timeout=60); checks["unit_tests"] = ut.returncode == 0
        files = [f"{f} ({b:,} bytes)" for f, b in info["files"].items()]; good = all(v is True for v in checks.values())
        text = (f"I built {spec['title']}: a full-stack app with a backend, a front end, a README, a Dockerfile and a test. {len(files)} files, {info['zip_bytes']:,} bytes zipped (the limit is 10 MB).\n\n"
                f"It keeps {spec['noun']} records with: {', '.join(x['label'] for x in spec['fields'])}. It is running now and I tried it like a browser would: " + ("everything worked" if good else "some checks failed: " + ", ".join(k for k, v in checks.items() if v is not True)) + ".")
        return {"kind": "action", "text": text, "files": files, "url": url, "zip": f"/download/{info['name']}.zip", "ms": (time.perf_counter() - t0) * 1000, "build": {k: v for k, v in info.items() if k != "files"}, "check": checks, "spec": spec}

    def chat(self, q):
        t0 = time.perf_counter(); low = q.lower()
        if re.search(r"\bstop\b.*\b(apps?|servers?|everything)\b|^\s*stop all", low):
            n = webapps.stop(); return {"kind": "action", "text": f"Stopped {n} app{'s' if n != 1 else ''}.", "ms": (time.perf_counter() - t0) * 1000}
        spec = webapps.parse(q)
        if spec and (not spec["recipe"] or re.search(r"full ?stack|backend|front ?end|download|deploy|zip|ready to|\bwith\b.+(?:,|\band\b)", low)):
            return self.fullstack(q, t0)
        if spec:
            if not spec["recipe"]:
                return {"kind": "told", "text": "I can build these small web apps: a to-do list, a notes app, a guestbook, a counter, a calculator or a stopwatch. Say which one, and you can add a colour and a name, for example: make a blue notes app called Study notes."}
            info = webapps.build(spec["recipe"], spec["title"], spec["color"])
            if not info["ok"]: return {"kind": "told", "text": "I could not start that app: " + info.get("why", "it did not answer")}
            chk = webapps.check_app(info); files = [f"{f} ({b:,} bytes)" for f, b in info["files"].items()]; r = webapps.RECIPES[info["recipe"]]
            text = (f"I built a {r['label']} called {info['title']}: {len(files)} files, {info['bytes']:,} bytes in total (the limit is 10 MB). It is running and I opened it as a browser would: "
                    f"{'the page loads' if chk['page_ok'] else 'the page did not load'}" + (f", and adding an item works" if chk.get("api_ok") else "") + f".\n\nHere is its main code:\n\n```{'python' if r['server'] == 'crud' else 'html'}\n{info['code']}\n```")
            return {"kind": "action", "text": text, "files": files, "url": info["url"], "ms": (time.perf_counter() - t0) * 1000, "build": {k: v for k, v in info.items() if k != "code"}, "check": chk}
        pref = "C " if re.search(r"(?<![a-z+#])c(?![a-z+#])", low) and "python" not in low else "Python " if "python" in low else None      # which language was asked for
        if pref:
            m = self.lesson(q, pref)
            if m: ms = (time.perf_counter() - t0) * 1000; self._agg("reply", ms); self._agg("reply_words", len(m["text"].split())); return {"kind": "chat", "text": m["text"], "ms": ms, "cat": m["cat"], "matched": m["matched"]}
        return super().chat(q)
