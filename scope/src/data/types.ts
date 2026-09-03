export type DataCategory =
  | 'identity' | 'contact' | 'location' | 'financial'
  | 'health' | 'preferences' | 'work' | 'secrets'

export const DATA_CATEGORIES: Record<DataCategory, { label: string; hint: string }> = {
  identity: { label: 'Identity', hint: 'Name, birth date, ID documents' },
  contact: { label: 'Contact', hint: 'Email addresses, phone numbers' },
  location: { label: 'Location', hint: 'Home address, places, travel plans' },
  financial: { label: 'Financial', hint: 'Bank details, cards, income' },
  health: { label: 'Health', hint: 'Conditions, allergies, prescriptions' },
  preferences: { label: 'Preferences', hint: 'History, tastes, devices' },
  work: { label: 'Work', hint: 'Employer, role, repositories' },
  secrets: { label: 'Secrets', hint: 'API keys, tokens, credentials' },
}

export type SourceKind = 'website' | 'codex' | 'claude-code' | 'browser'
export interface Source {
  id: string
  name: string
  kind: SourceKind
  lastSync: string
  color: string
}

export type Sensitivity = 'low' | 'medium' | 'high' | 'critical'
export interface DataItem {
  id: string
  label: string
  value: string
  category: DataCategory
  source: string
  sensitivity: Sensitivity
  collectedAt: string
}

export type AppCategoryId =
  | 'travel' | 'streaming' | 'shopping' | 'finance'
  | 'health' | 'productivity' | 'social' | 'developer'

export interface AppCategory { id: AppCategoryId; label: string; blurb: string }

export type Trust = 'verified' | 'community' | 'unknown'
export interface App {
  id: string
  name: string
  domain: string
  category: AppCategoryId
  blurb: string
  connected: boolean
  scopes: DataCategory[]
  trust: Trust
  color: string
  lastAccess?: string
  requests: number
}

export type AgentId = 'codex' | 'claude-code' | 'chatgpt' | 'cursor'
export const AGENTS: Record<AgentId, { label: string; short: string; cls: string }> = {
  codex: { label: 'Codex', short: 'CX', cls: 'codex' },
  'claude-code': { label: 'Claude Code', short: 'CC', cls: 'claude' },
  chatgpt: { label: 'ChatGPT agent', short: 'GP', cls: 'chatgpt' },
  cursor: { label: 'Cursor', short: 'CU', cls: 'cursor' },
}

export type Effect = 'allow' | 'ask' | 'deny'
export type RuleTarget =
  | { type: 'all' }
  | { type: 'category'; id: AppCategoryId }
  | { type: 'app'; id: string }
  | { type: 'agent'; id: AgentId }

export interface Rule {
  id: string
  effect: Effect
  data: DataCategory[] | 'all'
  target: RuleTarget
  enabled: boolean
  createdBy: 'user' | 'agent'
  note?: string
  createdAt: string
}

export type Decision = 'allowed' | 'denied' | 'asked' | 'blocked' | 'pending'
export interface ShareEvent {
  id: string
  at: string
  agent: AgentId
  session: string
  context: string
  app: string
  fields: string[]
  decision: Decision
  ruleId?: string
  flagged?: boolean
  reason?: string
}

export type Severity = 'critical' | 'warning' | 'info'
export interface Alert {
  id: string
  at: string
  severity: Severity
  title: string
  body: string
  eventId?: string
  resolved: boolean
}

export interface Scenario {
  id: string
  agent: AgentId
  session: string
  context: string
  app: string
  fields: string[]
}
