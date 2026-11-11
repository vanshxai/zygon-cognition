# Zygon.cognition

A small engineering assistant that runs in the browser. It is not a language model: a word-voting router picks the job, and exact calculators and a library of sourced formulas do the work. No GPU, no model download.

Two modes, chosen when you start a chat:
- **HVAC engineer**: duct and pipe sizes, cooling load, motor breaker, running cost, psychrometrics, invoices, maintenance reminders, any of ~200 physics/maths/engineering formulas.
- **Home electrician** (Indian homes): monthly bill, lights and fans per room, MCB and wire size, comparison with your real bill.

Every answer shows its steps. Inputs that break a physics rule (a negative flow, 200% humidity, a hot side colder than the cold side) are refused instead of calculated.

## Run it locally

    python3 -m http.server 5173      # then open http://localhost:5173

The chat sharing (below) needs the two API routes, so locally it simply stays quiet.

## Deploy on Vercel

1. Import this repository in Vercel. No build step, no framework: leave everything at the defaults.
2. **Optional, to collect the test chats:**
   - In the Vercel project go to **Storage**, add **Upstash Redis** (free tier) and connect it to the project. Vercel adds its environment variables by itself.
   - In **Settings → Environment Variables** add `ADMIN_KEY` with a long random value of your choice. Keep it private; it is the password for the admin page.
   - Redeploy once.
3. Open `https://<your-site>/admin`, type the admin key, and you can read every shared chat, search them, and download JSON or CSV.

## Test-version chat sharing

On first visit each person sees a notice. If they agree, each question they type and the title/first rows of Zygon's reply are sent to `/api/collect`, tagged with a random anonymous ID created in their browser. Email addresses and long digit strings are blanked first. Attached files are never sent. They can switch it off, and delete what was shared, in Settings.

- `api/collect.js` stores events in Redis (`log:<id>` lists, capped at 5,000 per person, rate limited).
- `api/export.js` returns them, only with the `x-admin-key` header.
- `admin.html` is the viewer. It is not linked from the app.

If you do not set up Redis, the app works exactly the same and nothing is sent.

## What is where

    index.html, style.css      the page
    js/app.js                  screen, chats, modes
    js/intake.js               cleans long, spoken or messy input down to the request
    js/engine.js               the voting router (Hexonet)
    js/tools.js, home.js       calculators for the two modes
    js/physics.js              refuses impossible inputs
    js/formulas.js             formula library solver (data/formulas.json, rules and sources)
    js/share.js                the opt-in chat sharing
    data/                      trained router tables and formulas
    api/, lib/                 the two Vercel functions
    vendor/                    pdf-lib, ExcelJS, PptxGenJS, pdf.js (permissive licences)

## Honest limits

- It handles the jobs above and nothing else. It does not chat or reason.
- Calculators are screening estimates; check them before ordering equipment.
- The router has only seen generated sentences so far. New phrasings can be misread, which is what the test chats are for.
- Speech input uses your browser's speech service, which may send audio to the internet; it asks first.
