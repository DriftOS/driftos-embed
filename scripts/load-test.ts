#!/usr/bin/env tsx

/**
 * Load Test Script - Simulates Production Traffic
 *
 * Generates varied conversations with realistic branching behavior:
 * - New conversations spawn continuously
 * - Topics drift naturally causing branches
 * - Some messages route back to previous topics
 * - Varied conversation lengths (3-20 messages)
 */

const API_URL = process.env.API_URL || 'http://localhost:3000';
const REQUESTS_PER_SECOND = parseInt(process.env.RPS || '30', 10);
const DURATION_MINUTES = parseInt(process.env.DURATION || '60', 10);
const RANDOMIZE = process.env.RANDOMIZE !== 'false';
const JITTER_PERCENT = parseFloat(process.env.JITTER || '30');

// Probability controls
const NEW_CONVERSATION_PROB = 0.15; // 15% chance to start fresh conversation
const TOPIC_DRIFT_PROB = 0.20; // 20% chance to drift to unrelated topic
const ROUTE_BACK_PROB = 0.10; // 10% chance to return to previous topic

interface Stats {
  total: number;
  success: number;
  errors: number;
  rateLimited: number;
  stays: number;
  routes: number;
  branches: number;
  avgLatency: number;
  latencies: number[];
  activeConversations: number;
}

const stats: Stats = {
  total: 0,
  success: 0,
  errors: 0,
  rateLimited: 0,
  stays: 0,
  routes: 0,
  branches: 0,
  avgLatency: 0,
  latencies: [],
  activeConversations: 0,
};

// Topic domains with varied content
const TOPIC_DOMAINS = {
  travel: {
    destinations: ['Japan', 'Italy', 'Thailand', 'Iceland', 'Peru', 'Morocco', 'New Zealand', 'Greece'],
    questions: [
      'What are the best places to visit in {dest}?',
      'How much should I budget for a 2-week trip to {dest}?',
      'What time of year is best to visit {dest}?',
      'Do I need a visa for {dest}?',
      'What local foods should I try in {dest}?',
      'Is {dest} safe for solo travelers?',
      'How do I get around in {dest}?',
      'What hotels do you recommend in {dest}?',
      'Any hidden gems in {dest} tourists don\'t know about?',
      'Should I book tours in advance for {dest}?',
    ],
  },
  tech: {
    topics: ['laptop', 'smartphone', 'smart home', 'coding setup', 'gaming PC', 'VR headset'],
    questions: [
      'What {topic} should I buy for work?',
      'How much RAM do I need for a {topic}?',
      'Mac or Windows for my {topic}?',
      'What\'s the best budget {topic} in 2024?',
      'Is the new {topic} worth upgrading to?',
      'How do I set up my {topic} properly?',
      'What accessories do I need for my {topic}?',
      'Can you compare different {topic} options?',
      'What specs matter most for a {topic}?',
      'How long will my {topic} last before needing replacement?',
    ],
  },
  cooking: {
    dishes: ['pasta carbonara', 'sushi', 'beef Wellington', 'Thai curry', 'French onion soup', 'tacos'],
    questions: [
      'What\'s the best recipe for {dish}?',
      'What ingredients do I need for {dish}?',
      'How long does it take to make {dish}?',
      'Any tips for making perfect {dish}?',
      'Can I make {dish} ahead of time?',
      'What wine pairs well with {dish}?',
      'How do I avoid common mistakes making {dish}?',
      'Is there a vegetarian version of {dish}?',
      'What equipment do I need for {dish}?',
      'Can I freeze leftover {dish}?',
    ],
  },
  fitness: {
    goals: ['lose weight', 'build muscle', 'run a marathon', 'improve flexibility', 'get stronger'],
    questions: [
      'What\'s the best workout plan to {goal}?',
      'How long will it take to {goal}?',
      'What should I eat to {goal}?',
      'How often should I exercise to {goal}?',
      'Do I need a gym membership to {goal}?',
      'What supplements help to {goal}?',
      'How do I stay motivated to {goal}?',
      'What mistakes should I avoid when trying to {goal}?',
      'Can I {goal} at home without equipment?',
      'How do I track progress when I {goal}?',
    ],
  },
  home: {
    projects: ['bathroom renovation', 'kitchen remodel', 'garden landscaping', 'home office setup', 'basement finishing'],
    questions: [
      'How much does a {project} cost?',
      'Should I DIY my {project} or hire a contractor?',
      'How long does a {project} typically take?',
      'What permits do I need for a {project}?',
      'What are common mistakes in a {project}?',
      'How do I plan my {project}?',
      'What materials are best for a {project}?',
      'Can I live in my house during a {project}?',
      'How do I find a good contractor for a {project}?',
      'What ROI can I expect from a {project}?',
    ],
  },
  finance: {
    topics: ['investing', 'retirement planning', 'budgeting', 'buying a house', 'paying off debt'],
    questions: [
      'How do I get started with {topic}?',
      'What\'s the best strategy for {topic}?',
      'How much should I allocate for {topic}?',
      'What mistakes should I avoid in {topic}?',
      'Is now a good time for {topic}?',
      'What tools help with {topic}?',
      'Should I hire an advisor for {topic}?',
      'How does {topic} affect my taxes?',
      'What\'s the timeline for {topic}?',
      'How do I balance {topic} with other goals?',
    ],
  },
};

const DOMAIN_KEYS = Object.keys(TOPIC_DOMAINS) as (keyof typeof TOPIC_DOMAINS)[];

interface ConversationState {
  branchId?: string;
  messageCount: number;
  maxMessages: number;
  currentDomain: keyof typeof TOPIC_DOMAINS;
  currentSubject: string;
  previousTopics: Array<{ domain: keyof typeof TOPIC_DOMAINS; subject: string }>;
}

// Active conversations
const conversations: Map<string, ConversationState> = new Map();
let conversationCounter = 0;

/**
 * Generate a random subject for a domain
 */
function getRandomSubject(domain: keyof typeof TOPIC_DOMAINS): string {
  const domainData = TOPIC_DOMAINS[domain];
  const subjects =
    'destinations' in domainData
      ? domainData.destinations
      : 'topics' in domainData
        ? domainData.topics
        : 'dishes' in domainData
          ? domainData.dishes
          : 'goals' in domainData
            ? domainData.goals
            : (domainData as { projects: string[] }).projects;
  return subjects[Math.floor(Math.random() * subjects.length)];
}

/**
 * Generate a message for a domain and subject
 */
function generateMessage(domain: keyof typeof TOPIC_DOMAINS, subject: string): string {
  const domainData = TOPIC_DOMAINS[domain];
  const questions = domainData.questions;
  const template = questions[Math.floor(Math.random() * questions.length)];

  // Replace placeholder with subject
  return template
    .replace('{dest}', subject)
    .replace('{topic}', subject)
    .replace('{dish}', subject)
    .replace('{goal}', subject)
    .replace('{project}', subject);
}

/**
 * Create a new conversation
 */
function createConversation(): string {
  conversationCounter++;
  const randomSuffix = Math.random().toString(36).substring(2, 8);
  const id = `conv-${Date.now()}-${conversationCounter}-${randomSuffix}`;
  const domain = DOMAIN_KEYS[Math.floor(Math.random() * DOMAIN_KEYS.length)];
  const subject = getRandomSubject(domain);

  conversations.set(id, {
    messageCount: 0,
    maxMessages: 3 + Math.floor(Math.random() * 18), // 3-20 messages
    currentDomain: domain,
    currentSubject: subject,
    previousTopics: [],
  });

  stats.activeConversations = conversations.size;
  return id;
}

/**
 * Get or create a conversation, handling lifecycle
 */
function getActiveConversation(): { id: string; state: ConversationState } {
  // Maybe start a new conversation
  if (conversations.size === 0 || Math.random() < NEW_CONVERSATION_PROB) {
    const id = createConversation();
    return { id, state: conversations.get(id)! };
  }

  // Pick a random existing conversation
  const ids = Array.from(conversations.keys());
  const id = ids[Math.floor(Math.random() * ids.length)];
  const state = conversations.get(id)!;

  // Check if conversation is exhausted
  if (state.messageCount >= state.maxMessages) {
    conversations.delete(id);
    stats.activeConversations = conversations.size;
    // Recurse to get another conversation
    return getActiveConversation();
  }

  return { id, state };
}

/**
 * Generate the next message for a conversation
 */
function getNextMessage(state: ConversationState): string {
  state.messageCount++;

  // First message is always about the initial topic
  if (state.messageCount === 1) {
    return `I want to ${state.currentDomain === 'travel' ? 'plan a trip to' : state.currentDomain === 'cooking' ? 'learn to make' : 'learn about'} ${state.currentSubject}`;
  }

  // Maybe drift to a completely different topic (causes BRANCH)
  if (Math.random() < TOPIC_DRIFT_PROB) {
    // Save current topic
    state.previousTopics.push({
      domain: state.currentDomain,
      subject: state.currentSubject,
    });

    // Pick a different domain
    const otherDomains = DOMAIN_KEYS.filter((d) => d !== state.currentDomain);
    state.currentDomain = otherDomains[Math.floor(Math.random() * otherDomains.length)];
    state.currentSubject = getRandomSubject(state.currentDomain);

    // Generate a topic-changing message
    const transitions = [
      `Actually, let me ask about something else - ${state.currentSubject}`,
      `Changing topics - I need help with ${state.currentSubject}`,
      `Quick question about ${state.currentSubject}`,
      `Totally different subject, but what about ${state.currentSubject}?`,
      `Oh wait, I also wanted to ask about ${state.currentSubject}`,
    ];
    return transitions[Math.floor(Math.random() * transitions.length)];
  }

  // Maybe route back to a previous topic (causes ROUTE)
  if (state.previousTopics.length > 0 && Math.random() < ROUTE_BACK_PROB) {
    const prevTopic = state.previousTopics[Math.floor(Math.random() * state.previousTopics.length)];
    state.currentDomain = prevTopic.domain;
    state.currentSubject = prevTopic.subject;

    const returns = [
      `Back to ${state.currentSubject} - `,
      `Going back to ${state.currentSubject}, `,
      `Returning to my question about ${state.currentSubject} - `,
      `About ${state.currentSubject} again - `,
    ];
    return returns[Math.floor(Math.random() * returns.length)] + generateMessage(state.currentDomain, state.currentSubject);
  }

  // Normal continuation (causes STAY)
  return generateMessage(state.currentDomain, state.currentSubject);
}

/**
 * Send a drift route request
 */
async function sendDriftRequest(): Promise<void> {
  const { id, state } = getActiveConversation();
  const content = getNextMessage(state);

  const start = Date.now();

  try {
    const response = await fetch(`${API_URL}/api/v1/drift/route`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        conversationId: id,
        content,
        role: 'user',
        currentBranchId: state.branchId,
      }),
    });

    const latency = Date.now() - start;
    stats.latencies.push(latency);
    stats.total++;

    if (response.ok) {
      const data = await response.json();
      stats.success++;

      // Track routing actions
      const action = data.data?.action;
      if (action === 'STAY') stats.stays++;
      else if (action === 'ROUTE') stats.routes++;
      else if (action === 'BRANCH') stats.branches++;

      // Update branch state for next request
      if (data.data?.branchId) {
        state.branchId = data.data.branchId;
      }
    } else if (response.status === 429) {
      stats.rateLimited++;
      stats.errors++;
    } else {
      stats.errors++;
      console.error(`Error ${response.status}: ${await response.text()}`);
    }
  } catch (error) {
    stats.total++;
    stats.errors++;
    console.error(`Request failed:`, error instanceof Error ? error.message : error);
  }
}

/**
 * Calculate and display stats
 */
function displayStats(): void {
  if (stats.latencies.length === 0) return;

  const sorted = [...stats.latencies].sort((a, b) => a - b);
  const avg = stats.latencies.reduce((a, b) => a + b, 0) / stats.latencies.length;
  const p50 = sorted[Math.floor(sorted.length * 0.5)];
  const p95 = sorted[Math.floor(sorted.length * 0.95)];
  const p99 = sorted[Math.floor(sorted.length * 0.99)];

  const successRate = ((stats.success / stats.total) * 100).toFixed(1);
  const errorRate = ((stats.errors / stats.total) * 100).toFixed(1);

  const totalActions = stats.stays + stats.routes + stats.branches;
  const stayPct = totalActions > 0 ? ((stats.stays / totalActions) * 100).toFixed(1) : '0.0';
  const routePct = totalActions > 0 ? ((stats.routes / totalActions) * 100).toFixed(1) : '0.0';
  const branchPct = totalActions > 0 ? ((stats.branches / totalActions) * 100).toFixed(1) : '0.0';

  console.clear();
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║                                                            ║');
  console.log('║   DRIFT LOAD TEST - REALISTIC CONVERSATIONS                ║');
  console.log('║                                                            ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  console.log('');
  console.log(`REQUESTS`);
  console.log(`   Total:       ${stats.total.toString().padStart(8)}`);
  console.log(`   Success:     ${stats.success.toString().padStart(8)} (${successRate}%)`);
  console.log(`   Errors:      ${stats.errors.toString().padStart(8)} (${errorRate}%)`);
  if (stats.rateLimited > 0) {
    console.log(`   Rate Limited:${stats.rateLimited.toString().padStart(8)}`);
  }
  console.log(`   Rate:        ${REQUESTS_PER_SECOND.toString().padStart(8)} req/s`);
  console.log('');
  console.log(`ROUTING ACTIONS`);
  console.log(`   STAY:        ${stats.stays.toString().padStart(8)} (${stayPct}%)`);
  console.log(`   ROUTE:       ${stats.routes.toString().padStart(8)} (${routePct}%)`);
  console.log(`   BRANCH:      ${stats.branches.toString().padStart(8)} (${branchPct}%)`);
  console.log('');
  console.log(`CONVERSATIONS`);
  console.log(`   Active:      ${stats.activeConversations.toString().padStart(8)}`);
  console.log(`   Total:       ${conversationCounter.toString().padStart(8)}`);
  console.log('');
  console.log(`LATENCY`);
  console.log(`   Avg:         ${avg.toFixed(1).padStart(8)} ms`);
  console.log(`   P50:         ${p50?.toString().padStart(8) ?? 'N/A'} ms`);
  console.log(`   P95:         ${p95?.toString().padStart(8) ?? 'N/A'} ms`);
  console.log(`   P99:         ${p99?.toString().padStart(8) ?? 'N/A'} ms`);
  console.log('');
  const grafanaPort = process.env.GRAFANA_PORT || '3002';
  console.log(`TARGET`);
  console.log(`   URL:         ${API_URL}/api/v1/drift/route`);
  console.log(`   Grafana:     http://localhost:${grafanaPort}`);
  console.log('');
  console.log(`Press Ctrl+C to stop`);
  console.log('');
}

/**
 * Calculate next request delay with jitter
 */
function getNextDelay(baseInterval: number): number {
  if (!RANDOMIZE) {
    return baseInterval;
  }

  const jitterAmount = baseInterval * (JITTER_PERCENT / 100);
  const minDelay = baseInterval - jitterAmount;
  const maxDelay = baseInterval + jitterAmount;

  return Math.random() * (maxDelay - minDelay) + minDelay;
}

/**
 * Main load test loop
 */
async function runLoadTest(): Promise<void> {
  console.log('Starting Drift load test...\n');

  console.log(`Target: ${API_URL}/api/v1/drift/route`);
  console.log(`Rate: ${REQUESTS_PER_SECOND} req/s`);
  console.log(`Randomized: ${RANDOMIZE ? `Yes (+-${JITTER_PERCENT}% jitter)` : 'No'}`);
  console.log(`Duration: ${DURATION_MINUTES} minutes`);
  console.log(`New conversation prob: ${(NEW_CONVERSATION_PROB * 100).toFixed(0)}%`);
  console.log(`Topic drift prob: ${(TOPIC_DRIFT_PROB * 100).toFixed(0)}%`);
  console.log(`Route back prob: ${(ROUTE_BACK_PROB * 100).toFixed(0)}%\n`);

  const baseInterval = 1000 / REQUESTS_PER_SECOND;
  const endTime = Date.now() + DURATION_MINUTES * 60 * 1000;

  // Display stats every second
  const statsInterval = setInterval(displayStats, 1000);

  // Recursive function for randomized intervals
  const scheduleNextRequest = () => {
    if (Date.now() >= endTime) {
      clearInterval(statsInterval);
      displayStats();
      console.log('\nLoad test complete!\n');
      process.exit(0);
      return;
    }

    // Send request
    sendDriftRequest().catch(console.error);

    // Schedule next request with jitter
    const nextDelay = getNextDelay(baseInterval);
    setTimeout(scheduleNextRequest, nextDelay);
  };

  // Start the first request
  scheduleNextRequest();

  // Handle Ctrl+C gracefully
  process.on('SIGINT', () => {
    clearInterval(statsInterval);
    displayStats();
    console.log('\nLoad test stopped by user\n');
    process.exit(0);
  });
}

// Start the load test
runLoadTest().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
