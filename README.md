# Zygon.cognition

A small engineering assistant that runs in the browser. It is not a language model: a word-voting router picks the job, and exact calculators and a library of sourced formulas do the work. No GPU, no model download.

Two modes, chosen when you start a chat:
- **HVAC engineer**: duct and pipe sizes, cooling load, motor breaker, running cost, psychrometrics, invoices, maintenance reminders, any of ~200 physics/maths/engineering formulas.
- **Home electrician** (Indian homes): monthly bill, lights and fans per room, MCB and wire size, comparison with your real bill.

Every answer shows its steps. Inputs that break a physics rule (a negative flow, 200% humidity, a hot side colder than the cold side) are refused instead of calculated.

## Run it locally

    python3 -m http.server 5173      # then open http://localhost:5173


## Deploy on Vercel

Import this repository in Vercel and deploy with the defaults. There is no build step, no environment variable and no database to set up: it is a static site.

## Test-version chat sharing

On first visit each person sees a notice and is asked for their name (nothing else). If they agree, their name, each question they type and the title/first rows of Zygon's reply are sent in small batches, tagged with a random anonymous ID created in their browser. Email addresses and long digit strings typed into messages are blanked first. Attached files are never sent. They can switch it off, change their name, and ask for deletion in Settings.

The batches go to a write-only Make.com webhook (`js/share.js`) that files them in a private data store. The webhook address can only be written to; the chats can be read only from the maintainer's Make account. Free Make plans allow about 1,000 actions and 1 MB of storage a month, which is enough for a small test.

If the webhook is unreachable the app works exactly the same and nothing is sent.

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
    vendor/                    pdf-lib, ExcelJS, PptxGenJS, pdf.js (permissive licences)

## Honest limits

- It handles the jobs above and nothing else. It does not chat or reason.
- Calculators are screening estimates; check them before ordering equipment.
- The router has only seen generated sentences so far. New phrasings can be misread, which is what the test chats are for.
- Speech input uses your browser's speech service, which may send audio to the internet; it asks first.
