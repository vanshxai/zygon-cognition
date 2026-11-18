"""Builds two offline reports:  software_index.html  (the 10 programs)  and  neuron_capabilities_report.html  (what the other neurons do)."""
import html, json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
WORK = os.path.join(ROOT, "model", "frost_work"); SW = os.path.join(ROOT, "software")
STYLE = ("<style>body{font:13px/1.4 Consolas,monospace;margin:14px;background:#fff;color:#000}pre{margin:2px;white-space:pre;overflow-x:auto}"
         "details{margin-left:1.2em}summary{cursor:pointer}h3{font:700 16px system-ui,sans-serif}</style>")


def det(title, body, op=False):
    return f"<details{' open' if op else ''}><summary>{html.escape(title)}</summary><pre>{html.escape(body)}</pre></details>"


def software_index():
    info = [("01_hello", "hello", "console", "Prints a greeting, the time and the machine name.", "double-click, or: hello.exe",
             "ran: prints greeting, date and machine"),
            ("02_calculator", "calc", "console", "Command-line calculator: + - * / % ^ ( ), sqrt sin cos tan abs ln log, pi, e. One-shot or interactive.", "calc.exe 2+3*4   or just calc.exe",
             "tested: 2+3*4=14, (1+2)^3/4+sqrt(16)=10.75, 10/0 -> clear error, sin(pi/2)+ln(e)=2"),
            ("03_todo", "todo", "console", "Todo list saved in todo.txt: add / list / done / undo / del / clear.", "todo.exe add Buy milk   |   todo.exe list",
             "tested: add, done, list, clear, bad index error, data survives between runs"),
            ("04_tictactoe", "tictactoe", "console", "Tic-tac-toe against an unbeatable computer (minimax).", "tictactoe.exe   (self-play check: tictactoe.exe --selftest)",
             "tested: computer vs computer always ends in a draw; human moves accepted"),
            ("05_notepad", "notepad", "Windows app", "Text editor: New/Open/Save/Save As, Find, word wrap, font, line/column/character status bar, unsaved-changes prompt.", "double-click notepad.exe",
             "tested: built-in self-test opens a file, finds text, saves it (OK)"),
            ("06_diskusage", "diskusage", "Windows app", "Disk usage analyzer: scans a folder in the background, lists the biggest files, totals per file type, exports CSV.", "double-click diskusage.exe",
             "tested: built-in self-test scans a folder, ranks files, writes CSV (OK)"),
            ("07_notes_server", "notes_server", "web app (backend + frontend + storage)", "Notes app: HTTP server with JSON API, browser page, notes.json storage.", "notes_server.exe   -> http://localhost:8851",
             "tested over HTTP: create, list, delete, validation (400), missing (404), file persistence"),
            ("08_taskboard", "taskboard", "web app (backend + frontend + storage)", "Kanban board: REST API, drag-and-drop page, search, stats, CSV export, request log, tasks.json.", "taskboard.exe   -> http://localhost:8852",
             "tested over HTTP: create, move columns, search, stats, CSV, validation errors 400/404, persistence"),
            ("09_chat_server", "chat_server", "web app (backend + frontend + storage)", "Multi-room chat: long polling, online users, /me and /shrug, search, rate limit, chat.log history.", "chat_server.exe   -> http://localhost:8853",
             "tested over HTTP: long poll answered ~1 s after a message, rate limit blocks the 7th message in 10 s, history survives restart"),
            ("10_miniwiki", "miniwiki", "web app (backend + frontend + storage)", "Wiki platform: accounts (salted PBKDF2), sessions, versions + revert, markdown, search, admin page, access log.", "miniwiki.exe   -> http://localhost:8854",
             "tested over HTTP: register/login, first user = admin, edit + 3 versions + revert, search, XSS escaping, CSRF header, roles (403), login rate limit (429), no plain passwords stored")]
    rows = []
    for folder, exe, kind, what, run, tested in info:
        d = os.path.join(SW, folder); cs = os.path.join(d, exe + ".cs" if exe != "notepad" or True else "")
        src = [f for f in os.listdir(d) if f.endswith(".cs")][0]; lines = len(open(os.path.join(d, src), encoding="utf-8").read().splitlines())
        kb = os.path.getsize(os.path.join(d, exe + ".exe")) / 1024
        rows.append(det(f"{folder}   {kind}   {lines} lines   {kb:.0f} KB", f"what it is : {what}\nexe        : {os.path.join(d, exe + '.exe')}\nsource     : {os.path.join(d, src)}\nrun        : {run}\n{tested}"))
    head = ("10 real, working Windows programs, smallest to biggest. Each is a single .exe built with the C# compiler that ships with Windows (no installs).\n"
            "They use only the .NET Framework that is already on this PC. Web apps open a page in your browser; stop them with Ctrl+C or by closing the window.\n"
            "Rebuild any program:  C:\\Windows\\Microsoft.NET\\Framework64\\v4.0.30319\\csc.exe /target:exe /r:System.Web.Extensions.dll /r:System.Core.dll file.cs\n"
            "IMPORTANT: I (not the Frost model) wrote these programs; Frost then read them to find which neurons fire (see the coding-neurons map).")
    page = f"<!doctype html><html><head><meta charset='utf-8'><title>Frost 1.0 - 10 programs</title>{STYLE}</head><body><h3>The 10 programs (small to big)</h3>{det('About', head, True)}{''.join(rows)}</body></html>"
    open(os.path.join(ROOT, "software_index.html"), "w", encoding="utf-8").write(page)
    print("software_index.html written")


def capabilities_report():
    r = json.load(open(os.path.join(WORK, "category_scan.json"), encoding="utf-8"))
    tot = r["total"]; cats = r["categories"]
    spec = sum(v["specific_neurons"] for v in cats.values())
    intro = f"""QUESTION: "only ~2% fire - what do the other 98% do?"

SHORT ANSWER: they are not idle and they do not have 'other jobs'. Two different 2% numbers were mixed up:
  - at any ONE token only about 2-3% of the 396,288 neurons are strongly on (the network is sparse),
  - but DIFFERENT tokens switch on DIFFERENT neurons, so over a varied set of replies {100 * r['ever_triggered_in_this_suite'] / tot:.1f}% of all neurons fired at least once
    ({r['ever_triggered_in_this_suite']:,} of {tot:,}) in just 38 short replies.

WHAT EVERY NEURON IS FOR (measured on 8 kinds of replies; 'busy' = in the busiest quarter of its layer)
  specific to ONE kind of reply ......... {spec:>8,}  ({100 * spec / tot:5.2f}%)
  busy in 5 or more of the 8 kinds ...... {r['shared_busy_everywhere']:>8,}  ({100 * r['shared_busy_everywhere'] / tot:5.2f}%)   general-purpose machinery
  in between (mixed roles) .............. {r['mixed']:>8,}  ({100 * r['mixed'] / tot:5.2f}%)
  never busy in any of the 8 kinds ...... {r['quiet_in_every_category']:>8,}  ({100 * r['quiet_in_every_category'] / tot:5.2f}%)   rarely used on these tests (they still fire occasionally)

So most neurons are NOT specialists: skills are spread across shared and mixed neurons. Only about 1% are clearly specific to one kind of reply."""
    spec_lines = "\n".join(f"  {c:13s} {v['specific_neurons']:>6,} neurons  ({100 * v['specific_neurons'] / tot:5.2f}%)   from {v['replies']} replies" for c, v in cats.items())
    A = r["ablation"]; L = A["loss"]; base = A["baseline"]
    groups = list(base)
    lines = ["Loss = average next-word error on 4 held-out replies per type (lower is better). Number in brackets = change vs normal.",
             "  switched off".ljust(34) + "".join(g.rjust(22) for g in groups), "  " + "-" * 120]
    for name, row in L.items():
        lines.append("  " + f"{name} ({A['sizes'].get(name, '-')})".ljust(32) + "".join(f"{row[g]:.3f} ({row[g] - base[g]:+.3f})".rjust(22) for g in groups))
    abl = "\n".join(lines)
    reading = f"""HOW TO READ IT
  - A group is 'real' if switching it off hurts ITS OWN kind of reply much more than switching off the same NUMBER of random neurons.
  - languages group (2,749 neurons): multilingual loss {base['multilingual']:.2f} -> {L['languages set']['multilingual']:.2f} ({L['languages set']['multilingual'] - base['multilingual']:+.2f}); random neurons of the same size: {L['RANDOM same size as text set']['multilingual'] - base['multilingual']:+.2f}. A clear specialist module.
  - text-reply group (2,874): text loss {L['text-reply set']['text'] - base['text']:+.2f} vs random {L['RANDOM same size as text set']['text'] - base['text']:+.2f}, but code only {L['text-reply set']['code'] - base['code']:+.2f} vs random {L['RANDOM same size as text set']['code'] - base['code']:+.2f}. Specific, and it protects code.
  - code group (753) and math group (314): small effects, close to random. Code and math lean on shared and mixed neurons rather than a private module.
  - Nothing breaks: even with a whole group switched off the model still answers fluently (see the generations). Skills are redundant and spread out.
  Limits: 4 held-out replies per type, one run, 'switched off' = activation set to 0. Treat the sizes of the effects as indicative."""
    gens = "\n".join(f"[{k}]\n   sea : {v[0]}\n   code: {v[1]}\n" for k, v in r["generations"].items())
    bl = r["by_layer"]
    layer_lines = []
    for c, arr in bl.items():
        layer_lines.append(det(f"{c}: where its {sum(arr):,} specific neurons sit (per layer)", "\n".join(f"layer {i:>2}  {n:>5}  {'#' * round(40 * n / max(max(arr), 1))}" for i, n in enumerate(arr))))
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>Frost 1.0 - what the other neurons do</title>{STYLE}</head><body><h3>Frost 1.0 - what do the other 98% of neurons do?</h3>"
            + det("1. Answer and the neuron census", intro, True) + det("2. Specific neurons per kind of reply", spec_lines)
            + det("3. Switch-off test (ablation): does removing a group hurt its own skill?", abl + "\n\n" + reading, True) + det("4. What the model writes with a group switched off", gens)
            + "<details><summary>5. Where the specific neurons sit, layer by layer</summary>" + "".join(layer_lines) + "</details></body></html>")
    open(os.path.join(ROOT, "neuron_capabilities_report.html"), "w", encoding="utf-8").write(page)
    print("neuron_capabilities_report.html written")


if __name__ == "__main__":
    software_index(); capabilities_report()
