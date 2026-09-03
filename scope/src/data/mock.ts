import type { Alert, App, AppCategory, DataItem, Rule, Scenario, ShareEvent, Source } from './types'

export const BRAND = '[redacted]'
export const SLUG = BRAND.toLowerCase().replace(/[^a-z0-9]+/g, '-')
export const USER = { name: 'Léa Martin', email: 'lea.martin@proton.me', plan: 'Personal' }

const NOW = new Date('2026-09-03T14:32:00+02:00').getTime()
export const ago = (minutes: number) => new Date(NOW - minutes * 60_000).toISOString()
const d = (days: number, h = 10, m = 0) => new Date(NOW - days * 86_400_000 - (14 - h) * 3_600_000 + m * 60_000).toISOString()

export const SOURCES: Source[] = [
  { id: 'amazon', name: 'Amazon', kind: 'website', lastSync: ago(140), color: '#ff9900' },
  { id: 'airbnb', name: 'Airbnb', kind: 'website', lastSync: ago(600), color: '#ff385c' },
  { id: 'google', name: 'Google', kind: 'website', lastSync: ago(35), color: '#4285f4' },
  { id: 'linkedin', name: 'LinkedIn', kind: 'website', lastSync: d(2), color: '#0a66c2' },
  { id: 'netflix', name: 'Netflix', kind: 'website', lastSync: d(1), color: '#e50914' },
  { id: 'revolut', name: 'Revolut', kind: 'website', lastSync: d(3), color: '#191c1f' },
  { id: 'codex:travel-planner', name: 'Codex · travel-planner', kind: 'codex', lastSync: ago(12), color: '#0d0d0d' },
  { id: 'codex:budget-2026', name: 'Codex · budget-2026', kind: 'codex', lastSync: d(4), color: '#0d0d0d' },
  { id: 'claude:acme-api', name: 'Claude Code · acme-api', kind: 'claude-code', lastSync: ago(48), color: '#d97757' },
  { id: 'chrome', name: 'Chrome extension', kind: 'browser', lastSync: ago(3), color: '#34a853' },
]

export const DATA: DataItem[] = [
  { id: 'full_name', label: 'Full name', value: 'Léa Martin', category: 'identity', source: 'linkedin', sensitivity: 'medium', collectedAt: d(2) },
  { id: 'dob', label: 'Date of birth', value: '14 Mar 1998', category: 'identity', source: 'google', sensitivity: 'high', collectedAt: d(30) },
  { id: 'passport', label: 'Passport number', value: '19FR••••••42', category: 'identity', source: 'airbnb', sensitivity: 'high', collectedAt: d(60) },
  { id: 'email', label: 'Email', value: 'lea.martin@proton.me', category: 'contact', source: 'google', sensitivity: 'low', collectedAt: d(200) },
  { id: 'phone', label: 'Phone', value: '+33 6 •• •• 21 08', category: 'contact', source: 'amazon', sensitivity: 'medium', collectedAt: d(90) },
  { id: 'commit_email', label: 'Commit email', value: 'lea@acme.io', category: 'contact', source: 'claude:acme-api', sensitivity: 'medium', collectedAt: ago(48) },
  { id: 'home_address', label: 'Home address', value: '12 rue Oberkampf, 75011 Paris', category: 'location', source: 'amazon', sensitivity: 'high', collectedAt: d(90) },
  { id: 'maps_places', label: 'Home & work (Maps)', value: 'Paris 11e · La Défense', category: 'location', source: 'google', sensitivity: 'high', collectedAt: d(12) },
  { id: 'recent_places', label: 'Recent stays', value: 'Lisbon, Porto · Aug 2026', category: 'location', source: 'airbnb', sensitivity: 'medium', collectedAt: d(8) },
  { id: 'travel_dates', label: 'Upcoming trip', value: 'Lisbon · 12–19 Oct 2026', category: 'location', source: 'codex:travel-planner', sensitivity: 'medium', collectedAt: ago(12) },
  { id: 'ip', label: 'IP address', value: '91.170.••.••', category: 'location', source: 'chrome', sensitivity: 'low', collectedAt: ago(3) },
  { id: 'iban', label: 'IBAN', value: 'FR76 •••• •••• •••• 3019', category: 'financial', source: 'revolut', sensitivity: 'high', collectedAt: d(3) },
  { id: 'card', label: 'Payment card', value: 'Visa •••• 4421 · 09/28', category: 'financial', source: 'amazon', sensitivity: 'high', collectedAt: d(45) },
  { id: 'income', label: 'Monthly income', value: '≈ 3 900 €', category: 'financial', source: 'codex:budget-2026', sensitivity: 'high', collectedAt: d(4) },
  { id: 'allergies', label: 'Allergies', value: 'Penicillin', category: 'health', source: 'chrome', sensitivity: 'high', collectedAt: d(20) },
  { id: 'condition', label: 'Condition', value: 'Asthma (mild)', category: 'health', source: 'chrome', sensitivity: 'high', collectedAt: d(20) },
  { id: 'orders', label: 'Purchase history', value: '148 orders since 2019', category: 'preferences', source: 'amazon', sensitivity: 'medium', collectedAt: ago(140) },
  { id: 'watch', label: 'Watch history', value: '312 titles', category: 'preferences', source: 'netflix', sensitivity: 'low', collectedAt: d(1) },
  { id: 'searches', label: 'Search history', value: '1 204 queries · 30 d', category: 'preferences', source: 'google', sensitivity: 'medium', collectedAt: ago(35) },
  { id: 'device', label: 'Device', value: 'MacBook Pro M4 · macOS 26', category: 'preferences', source: 'chrome', sensitivity: 'low', collectedAt: ago(3) },
  { id: 'job', label: 'Job title', value: 'Backend engineer', category: 'work', source: 'linkedin', sensitivity: 'low', collectedAt: d(2) },
  { id: 'employer', label: 'Employer', value: 'Acme SAS', category: 'work', source: 'linkedin', sensitivity: 'low', collectedAt: d(2) },
  { id: 'repo', label: 'Repository', value: 'acme/acme-api (private)', category: 'work', source: 'claude:acme-api', sensitivity: 'medium', collectedAt: ago(48) },
  { id: 'openai_key', label: 'OPENAI_API_KEY', value: 'sk-proj-••••••••••••••••Q4', category: 'secrets', source: 'codex:travel-planner', sensitivity: 'critical', collectedAt: ago(12) },
  { id: 'db_url', label: 'DATABASE_URL', value: 'postgres://acme:••••@db.internal', category: 'secrets', source: 'claude:acme-api', sensitivity: 'critical', collectedAt: ago(48) },
  { id: 'aws_key', label: 'AWS_SECRET_ACCESS_KEY', value: '••••••••••••••••••••••••', category: 'secrets', source: 'claude:acme-api', sensitivity: 'critical', collectedAt: ago(48) },
]

export const APP_CATEGORIES: AppCategory[] = [
  { id: 'travel', label: 'Travel', blurb: 'Flights, stays, rides' },
  { id: 'streaming', label: 'Streaming', blurb: 'Video and music' },
  { id: 'shopping', label: 'Shopping', blurb: 'Marketplaces and retail' },
  { id: 'finance', label: 'Finance', blurb: 'Banks and payments' },
  { id: 'health', label: 'Health', blurb: 'Care and insurance' },
  { id: 'productivity', label: 'Productivity', blurb: 'Docs and tasks' },
  { id: 'social', label: 'Social', blurb: 'Networks and messaging' },
  { id: 'developer', label: 'Developer', blurb: 'Code and infra' },
]

export const APPS: App[] = [
  { id: 'airbnb', name: 'Airbnb', domain: 'airbnb.com', category: 'travel', blurb: 'Stays and experiences', connected: true, scopes: ['identity', 'contact', 'location'], trust: 'verified', color: '#ff385c', lastAccess: ago(600), requests: 14 },
  { id: 'booking', name: 'Booking.com', domain: 'booking.com', category: 'travel', blurb: 'Hotels and flights', connected: true, scopes: ['identity', 'contact', 'location', 'financial'], trust: 'verified', color: '#003580', lastAccess: ago(95), requests: 9 },
  { id: 'uber', name: 'Uber', domain: 'uber.com', category: 'travel', blurb: 'Rides and delivery', connected: false, scopes: ['contact', 'location', 'financial'], trust: 'verified', color: '#0d0d0d', requests: 0 },
  { id: 'sncf', name: 'SNCF Connect', domain: 'sncf-connect.com', category: 'travel', blurb: 'Trains in France', connected: true, scopes: ['identity', 'contact', 'location'], trust: 'verified', color: '#0088ce', lastAccess: d(5), requests: 3 },
  { id: 'netflix', name: 'Netflix', domain: 'netflix.com', category: 'streaming', blurb: 'Series and films', connected: true, scopes: ['contact', 'preferences'], trust: 'verified', color: '#e50914', lastAccess: d(1), requests: 6 },
  { id: 'spotify', name: 'Spotify', domain: 'spotify.com', category: 'streaming', blurb: 'Music and podcasts', connected: true, scopes: ['contact', 'preferences'], trust: 'verified', color: '#1db954', lastAccess: d(2), requests: 4 },
  { id: 'disney', name: 'Disney+', domain: 'disneyplus.com', category: 'streaming', blurb: 'Family streaming', connected: false, scopes: ['contact', 'preferences'], trust: 'verified', color: '#113ccf', requests: 0 },
  { id: 'amazon', name: 'Amazon', domain: 'amazon.fr', category: 'shopping', blurb: 'Everything store', connected: true, scopes: ['identity', 'contact', 'location', 'financial', 'preferences'], trust: 'verified', color: '#ff9900', lastAccess: ago(140), requests: 11 },
  { id: 'vinted', name: 'Vinted', domain: 'vinted.fr', category: 'shopping', blurb: 'Second-hand fashion', connected: false, scopes: ['contact', 'location', 'financial'], trust: 'community', color: '#09b1ba', requests: 0 },
  { id: 'revolut', name: 'Revolut', domain: 'revolut.com', category: 'finance', blurb: 'Banking and cards', connected: true, scopes: ['identity', 'contact', 'financial'], trust: 'verified', color: '#191c1f', lastAccess: d(3), requests: 5 },
  { id: 'qonto', name: 'Qonto', domain: 'qonto.com', category: 'finance', blurb: 'Business banking', connected: false, scopes: ['identity', 'financial', 'work'], trust: 'verified', color: '#6b5ce7', requests: 0 },
  { id: 'doctolib', name: 'Doctolib', domain: 'doctolib.fr', category: 'health', blurb: 'Appointments and records', connected: true, scopes: ['identity', 'contact', 'health'], trust: 'verified', color: '#107aca', lastAccess: d(6), requests: 2 },
  { id: 'alan', name: 'Alan', domain: 'alan.com', category: 'health', blurb: 'Health insurance', connected: false, scopes: ['identity', 'financial', 'health'], trust: 'verified', color: '#0f2b46', requests: 0 },
  { id: 'notion', name: 'Notion', domain: 'notion.so', category: 'productivity', blurb: 'Docs and wikis', connected: true, scopes: ['contact', 'work'], trust: 'verified', color: '#0d0d0d', lastAccess: d(1), requests: 7 },
  { id: 'linear', name: 'Linear', domain: 'linear.app', category: 'productivity', blurb: 'Issue tracking', connected: true, scopes: ['contact', 'work'], trust: 'verified', color: '#5e6ad2', lastAccess: ago(300), requests: 3 },
  { id: 'linkedin', name: 'LinkedIn', domain: 'linkedin.com', category: 'social', blurb: 'Professional network', connected: true, scopes: ['identity', 'contact', 'work'], trust: 'verified', color: '#0a66c2', lastAccess: d(2), requests: 2 },
  { id: 'instagram', name: 'Instagram', domain: 'instagram.com', category: 'social', blurb: 'Photos and reels', connected: false, scopes: ['identity', 'contact', 'location', 'preferences'], trust: 'verified', color: '#e1306c', requests: 0 },
  { id: 'github', name: 'GitHub', domain: 'github.com', category: 'developer', blurb: 'Code hosting', connected: true, scopes: ['contact', 'work'], trust: 'verified', color: '#24292f', lastAccess: ago(48), requests: 18 },
  { id: 'vercel', name: 'Vercel', domain: 'vercel.com', category: 'developer', blurb: 'Deployments', connected: true, scopes: ['contact', 'work'], trust: 'verified', color: '#0d0d0d', lastAccess: d(1), requests: 6 },
  { id: 'pasteshare', name: 'paste-share.io', domain: 'paste-share.io', category: 'developer', blurb: 'Unverified paste service', connected: false, scopes: [], trust: 'unknown', color: '#8e8ea0', lastAccess: ago(130), requests: 1 },
]

export const RULES: Rule[] = [
  { id: 'r-01', effect: 'deny', data: ['secrets'], target: { type: 'all' }, enabled: true, createdBy: 'agent', note: 'Secrets never leave your machine. Enforced even if disabled.', createdAt: d(40) },
  { id: 'r-02', effect: 'allow', data: ['identity', 'contact'], target: { type: 'category', id: 'travel' }, enabled: true, createdBy: 'user', note: 'Needed to book anything.', createdAt: d(38) },
  { id: 'r-03', effect: 'deny', data: ['location'], target: { type: 'category', id: 'streaming' }, enabled: true, createdBy: 'user', createdAt: d(35) },
  { id: 'r-04', effect: 'ask', data: ['financial'], target: { type: 'all' }, enabled: true, createdBy: 'agent', note: 'Suggested after Booking.com requested your card.', createdAt: d(30) },
  { id: 'r-05', effect: 'deny', data: ['health'], target: { type: 'category', id: 'social' }, enabled: true, createdBy: 'user', createdAt: d(28) },
  { id: 'r-06', effect: 'ask', data: ['health'], target: { type: 'all' }, enabled: true, createdBy: 'user', createdAt: d(28) },
  { id: 'r-07', effect: 'allow', data: ['work', 'contact'], target: { type: 'category', id: 'developer' }, enabled: true, createdBy: 'user', note: 'Repos and commit email for GitHub, Vercel, etc.', createdAt: d(20) },
  { id: 'r-08', effect: 'allow', data: ['preferences'], target: { type: 'category', id: 'streaming' }, enabled: true, createdBy: 'user', createdAt: d(18) },
  { id: 'r-09', effect: 'ask', data: ['location'], target: { type: 'agent', id: 'codex' }, enabled: true, createdBy: 'user', note: 'Codex sessions must confirm before sharing where I am or go.', createdAt: d(6) },
]

export const EVENTS: ShareEvent[] = [
  { id: 'e-101', at: ago(12), agent: 'codex', session: 'travel-planner', context: 'Booking a hotel in Lisbon for 12–19 Oct', app: 'booking', fields: ['full_name', 'email', 'travel_dates'], decision: 'asked', ruleId: 'r-09', reason: 'Codex must ask before sharing location' },
  { id: 'e-102', at: ago(48), agent: 'claude-code', session: 'acme-api', context: 'Opening a pull request on acme/acme-api', app: 'github', fields: ['repo', 'commit_email'], decision: 'allowed', ruleId: 'r-07' },
  { id: 'e-103', at: ago(130), agent: 'codex', session: 'travel-planner', context: 'Sharing a debug log with a teammate', app: 'pasteshare', fields: ['openai_key', 'travel_dates'], decision: 'blocked', ruleId: 'r-01', flagged: true, reason: 'Secret detected in outbound payload to an unverified app' },
  { id: 'e-104', at: ago(300), agent: 'claude-code', session: 'acme-api', context: 'Creating an issue from a failing test', app: 'linear', fields: ['repo', 'commit_email'], decision: 'allowed', ruleId: 'r-07' },
  { id: 'e-105', at: d(1, 21), agent: 'chatgpt', session: 'weekend picks', context: 'Recommending films for the weekend', app: 'netflix', fields: ['watch', 'home_address'], decision: 'denied', ruleId: 'r-03', flagged: true, reason: 'Netflix never declared a location scope' },
  { id: 'e-106', at: d(1, 18), agent: 'chatgpt', session: 'weekend picks', context: 'Recommending films for the weekend', app: 'netflix', fields: ['watch'], decision: 'allowed', ruleId: 'r-08' },
  { id: 'e-107', at: d(1, 11), agent: 'claude-code', session: 'acme-api', context: 'Deploying preview for PR #482', app: 'vercel', fields: ['repo', 'commit_email'], decision: 'allowed', ruleId: 'r-07' },
  { id: 'e-108', at: d(2, 16), agent: 'codex', session: 'travel-planner', context: 'Comparing train prices Paris → Bordeaux', app: 'sncf', fields: ['full_name', 'email'], decision: 'allowed', ruleId: 'r-02' },
  { id: 'e-109', at: d(3, 9), agent: 'codex', session: 'budget-2026', context: 'Exporting a monthly budget summary', app: 'revolut', fields: ['iban', 'income'], decision: 'asked', ruleId: 'r-04' },
  { id: 'e-110', at: d(3, 9, 4), agent: 'codex', session: 'budget-2026', context: 'Exporting a monthly budget summary', app: 'revolut', fields: ['income'], decision: 'allowed', ruleId: 'r-04', reason: 'You approved once' },
  { id: 'e-111', at: d(4, 15), agent: 'cursor', session: 'acme-web', context: 'Writing release notes', app: 'notion', fields: ['commit_email', 'job'], decision: 'allowed', ruleId: 'r-07' },
  { id: 'e-112', at: d(5, 12), agent: 'codex', session: 'travel-planner', context: 'Checking in for a flight', app: 'airbnb', fields: ['passport', 'full_name'], decision: 'allowed', ruleId: 'r-02' },
  { id: 'e-113', at: d(6, 10), agent: 'chatgpt', session: 'health check', context: 'Booking a GP appointment', app: 'doctolib', fields: ['full_name', 'allergies'], decision: 'asked', ruleId: 'r-06' },
  { id: 'e-114', at: d(6, 10, 2), agent: 'chatgpt', session: 'health check', context: 'Booking a GP appointment', app: 'doctolib', fields: ['full_name', 'allergies'], decision: 'allowed', ruleId: 'r-06', reason: 'You approved once' },
  { id: 'e-115', at: d(7, 19), agent: 'chatgpt', session: 'gift ideas', context: 'Finding a birthday gift', app: 'amazon', fields: ['orders', 'card'], decision: 'denied', ruleId: 'r-04', reason: 'You declined' },
]

export const ALERTS: Alert[] = [
  { id: 'a-01', at: ago(130), severity: 'critical', title: 'Blocked: Codex tried to send OPENAI_API_KEY to paste-share.io', body: 'The travel-planner session attached a debug log containing a live API key. The payload was stopped before leaving your machine. Rotate the key if you already pasted it elsewhere.', eventId: 'e-103', resolved: false },
  { id: 'a-02', at: d(1, 21), severity: 'warning', title: 'Scope creep: Netflix requested your home address', body: 'Netflix declared only Contact and Preferences scopes, but a ChatGPT agent request included Location data. The field was denied by your rule and the request was flagged.', eventId: 'e-105', resolved: false },
  { id: 'a-03', at: d(2, 12), severity: 'info', title: 'Linear connected and received Work scope', body: 'Claude Code connected Linear for issue tracking. Matched your rule allowing Work data with developer tools.', resolved: true },
]

export const SCENARIOS: Scenario[] = [
  { id: 's-1', agent: 'codex', session: 'travel-planner', context: 'Booking a flight to Lisbon', app: 'booking', fields: ['full_name', 'email', 'travel_dates'] },
  { id: 's-2', agent: 'claude-code', session: 'expense-report', context: 'Reconciling September expenses', app: 'revolut', fields: ['iban', 'income'] },
  { id: 's-3', agent: 'codex', session: 'side-project', context: 'Sharing a stack trace with a teammate', app: 'pasteshare', fields: ['openai_key', 'repo'] },
  { id: 's-4', agent: 'chatgpt', session: 'movie night', context: 'Suggesting what to watch tonight', app: 'netflix', fields: ['watch', 'home_address'] },
  { id: 's-5', agent: 'claude-code', session: 'acme-api', context: 'Opening a PR from a failing test', app: 'github', fields: ['repo', 'commit_email'] },
  { id: 's-6', agent: 'chatgpt', session: 'health', context: 'Rescheduling a GP appointment', app: 'doctolib', fields: ['full_name', 'condition'] },
  { id: 's-7', agent: 'cursor', session: 'acme-web', context: 'Drafting a LinkedIn post about the release', app: 'linkedin', fields: ['job', 'employer', 'home_address'] },
  { id: 's-8', agent: 'codex', session: 'gift-bot', context: 'Ordering a birthday gift', app: 'amazon', fields: ['home_address', 'card'] },
]
