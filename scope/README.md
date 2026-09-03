# Scope — the OAuth layer for your data and AI agents

Hackathon demo built for the OpenAI AI Privacy Hackathon (Station F, Paris, 3 Sept 2026).
Front-end only, all data is mock. Styled after openai.com: white ground, near-black ink,
hairline borders, pill buttons, green only for "allowed", red only for "blocked".

## Run

Deployed by the jacques-blog GitHub Pages workflow at https://jacqkues.github.io/scope/ .
Locally:

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check + production bundle in dist/
```

## Rename the product

Change `BRAND` in `src/data/mock.ts`. Every visible label, the tab title and the
`@<slug>/consent` / `api.<slug>.dev` strings in the developer section derive from it.

## What the demo shows

| Route            | What it is                                                                                |
| ---------------- | ----------------------------------------------------------------------------------------- |
| `/`              | Landing page: hero with animated data → gate → apps flow, how it works, features, plugin snippet |
| `/app`           | Overview: stats, recent activity, open alerts, agent suggestion                            |
| `/app/data`      | Personal data vault with provenance (websites, Codex / Claude Code sessions, browser)     |
| `/app/apps`      | App catalogue by category, declared scopes coloured by the effective policy, "Explain" drawer |
| `/app/rules`     | Plain-language rule composer (mock NLU), rule list, agent suggestions                      |
| `/app/activity`  | Share log: agent, session, app, fields, decision, rule; expand for the raw plugin request |
| `/app/alerts`    | Critical / warning / info alerts with resolve actions                                      |

**"Simulate agent request"** (top bar) cycles through the scenarios in `SCENARIOS`
(`src/data/mock.ts`). Each one runs through the real decision engine:

1. Booking.com asks for name, email, travel dates → consent prompt (Codex must ask for location).
2. Revolut asks for IBAN + income → consent prompt; "Always allow" creates an app-specific rule.
3. paste-share.io receives an `OPENAI_API_KEY` → blocked, red takeover, critical alert, banner.
4. Netflix asks for watch history + home address → partial share, scope-creep warning.
5. …and four more (GitHub, Doctolib, LinkedIn, Amazon).

## Where the logic lives

- `src/store/engine.ts` — `evaluate()` (per-field verdicts, secrets hard-block, scope creep,
  rule specificity app › agent › category › all, stricter effect wins on ties),
  `parseRule()` (keyword NLU for the composer), `explainApp()` (the agent explanation text).
- `src/store/store.tsx` — React reducer holding apps, rules, events, alerts, the pending
  consent prompt, the critical takeover and toasts.
- `src/data/mock.ts` — all mock data: sources, data items, app categories, apps, rules,
  events, alerts, scenarios.

## Hooking the real plugin later

The consent flow is already shaped like the endpoint shown on the landing page:
`{ agent, session, app, fields }` in, `{ decision, rule, prompt }` out. Replace the
`simulate` action in `store.tsx` with a WebSocket / SSE subscription that dispatches the
same shape, and the prompt, log, alert and toast UI keep working unchanged.
