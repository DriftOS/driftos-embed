#!/usr/bin/env npx tsx
/**
 * End-to-end test using annotated conversation scenarios
 *
 * Loads manual_test_cases.json and replays conversations through the API,
 * validating drift detection matches expected annotations.
 *
 * Usage:
 *   npx tsx scripts/test-scenarios.ts
 *   npx tsx scripts/test-scenarios.ts --data /path/to/test_cases.json
 *   npx tsx scripts/test-scenarios.ts --scenario semantic_gradual
 */

import fs from 'fs';
import path from 'path';

const API_URL = process.env.API_URL || 'http://localhost:3001/api/v1';

// ANSI colors
const colors = {
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m',
  reset: '\x1b[0m',
  bold: '\x1b[1m',
};

interface Message {
  role: 'user' | 'assistant';
  content: string;
  mode?: string;
  topics?: string[];
}

interface Annotation {
  message_index: number;
  drift_type: string;
  should_branch: boolean;
  expected_mode?: string;
  notes?: string;
}

interface Conversation {
  id: string;
  scenario: string;
  messages: Message[];
  annotations: Annotation[];
  metadata?: {
    type?: string;
    purpose?: string;
    expected_branches?: number;
  };
}

interface TestData {
  version: string;
  conversations: Conversation[];
}

interface TestResult {
  scenario: string;
  conversationId: string;
  totalMessages: number;
  annotatedMessages: number;
  passed: number;
  failed: number;
  details: {
    messageIndex: number;
    content: string;
    expectedBranch: boolean;
    actualAction: string;
    similarity: number;
    passed: boolean;
    notes: string;
  }[];
}

async function sendMessage(
  conversationId: string,
  content: string,
  branchId?: string
): Promise<{
  success: boolean;
  action?: string;
  driftAction?: string;
  branchId?: string;
  similarity?: number;
  isNewBranch?: boolean;
  error?: string;
}> {
  const payload: Record<string, string> = {
    conversationId,
    content,
  };

  if (branchId) {
    payload.currentBranchId = branchId;
  }

  try {
    const response = await fetch(`${API_URL}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!data.success) {
      return { success: false, error: data.error?.message || 'Unknown error' };
    }

    return {
      success: true,
      action: data.data.action,
      driftAction: data.data.driftAction,
      branchId: data.data.branchId,
      similarity: data.data.similarity,
      isNewBranch: data.data.isNewBranch,
    };
  } catch (error) {
    return { success: false, error: String(error) };
  }
}

function shouldHaveBranched(driftAction: string): boolean {
  return driftAction === 'BRANCH_SAME_CLUSTER' || driftAction === 'BRANCH_NEW_CLUSTER';
}

async function runConversation(conversation: Conversation): Promise<TestResult> {
  const result: TestResult = {
    scenario: conversation.scenario,
    conversationId: `test-${conversation.id}-${Date.now()}`,
    totalMessages: conversation.messages.filter(m => m.role === 'user').length,
    annotatedMessages: conversation.annotations.length,
    passed: 0,
    failed: 0,
    details: [],
  };

  const annotationMap = new Map<number, Annotation>();
  for (const ann of conversation.annotations) {
    annotationMap.set(ann.message_index, ann);
  }

  let currentBranchId: string | undefined;

  console.log(`\n${colors.cyan}━━━ ${conversation.scenario}: ${conversation.id} ━━━${colors.reset}`);
  if (conversation.metadata?.purpose) {
    console.log(`    Purpose: ${conversation.metadata.purpose}`);
  }
  console.log();

  for (let i = 0; i < conversation.messages.length; i++) {
    const message = conversation.messages[i];

    // Skip assistant messages - only send user messages
    if (message.role === 'assistant') continue;

    const annotation = annotationMap.get(i);
    const shortContent = message.content.length > 60
      ? message.content.slice(0, 60) + '...'
      : message.content;

    console.log(`  ${colors.blue}[${i}]${colors.reset} "${shortContent}"`);

    const response = await sendMessage(
      result.conversationId,
      message.content,
      currentBranchId
    );

    if (!response.success) {
      console.log(`      ${colors.red}✗ Error: ${response.error}${colors.reset}`);
      result.failed++;
      continue;
    }

    // Update branch ID for next message
    currentBranchId = response.branchId;

    const actionColor = response.driftAction === 'STAY'
      ? colors.green
      : response.driftAction === 'BRANCH_SAME_CLUSTER'
        ? colors.yellow
        : colors.red;

    console.log(`      ${actionColor}▶ ${response.driftAction}${colors.reset} (sim=${response.similarity?.toFixed(3)})`);

    // Check against annotation if present
    if (annotation) {
      const actualBranched = shouldHaveBranched(response.driftAction || '');
      const matched = annotation.should_branch === actualBranched;

      result.details.push({
        messageIndex: i,
        content: shortContent,
        expectedBranch: annotation.should_branch,
        actualAction: response.driftAction || 'unknown',
        similarity: response.similarity || 0,
        passed: matched,
        notes: annotation.notes || '',
      });

      if (matched) {
        result.passed++;
        console.log(`      ${colors.green}✓ Expected: branch=${annotation.should_branch}${colors.reset}`);
      } else {
        result.failed++;
        console.log(`      ${colors.red}✗ Expected: branch=${annotation.should_branch}, got: ${response.driftAction}${colors.reset}`);
      }

      if (annotation.notes) {
        console.log(`      ${colors.magenta}📝 ${annotation.notes}${colors.reset}`);
      }
    }

    console.log();
  }

  return result;
}

async function main() {
  const args = process.argv.slice(2);

  // Parse arguments
  let dataPath = '../../DriftOS/data/drift-test-data/manual_test_cases.json';
  let scenarioFilter: string | undefined;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--data' && args[i + 1]) {
      dataPath = args[++i];
    } else if (args[i] === '--scenario' && args[i + 1]) {
      scenarioFilter = args[++i];
    }
  }

  // Resolve path relative to script location
  const resolvedPath = path.isAbsolute(dataPath)
    ? dataPath
    : path.resolve(import.meta.dirname || process.cwd(), dataPath);

  console.log(`${colors.bold}DriftOS Scenario Test Runner${colors.reset}`);
  console.log(`API: ${API_URL}`);
  console.log(`Data: ${resolvedPath}`);
  if (scenarioFilter) {
    console.log(`Filter: ${scenarioFilter}`);
  }

  // Load test data
  if (!fs.existsSync(resolvedPath)) {
    console.error(`${colors.red}Error: Test data not found at ${resolvedPath}${colors.reset}`);
    process.exit(1);
  }

  const testData: TestData = JSON.parse(fs.readFileSync(resolvedPath, 'utf-8'));
  console.log(`Loaded ${testData.conversations.length} conversations`);

  // Filter if needed
  let conversations = testData.conversations;
  if (scenarioFilter) {
    conversations = conversations.filter(c => c.scenario === scenarioFilter);
    console.log(`Filtered to ${conversations.length} conversations`);
  }

  // Run each conversation
  const results: TestResult[] = [];

  for (const conversation of conversations) {
    const result = await runConversation(conversation);
    results.push(result);
  }

  // Summary
  console.log(`\n${colors.bold}${colors.magenta}═══════════════════════════════════════════════════════════════${colors.reset}`);
  console.log(`${colors.bold}${colors.magenta}  SUMMARY${colors.reset}`);
  console.log(`${colors.bold}${colors.magenta}═══════════════════════════════════════════════════════════════${colors.reset}\n`);

  let totalPassed = 0;
  let totalFailed = 0;

  for (const result of results) {
    const icon = result.failed === 0 ? colors.green + '✓' : colors.red + '✗';
    console.log(`${icon}${colors.reset} ${result.scenario} (${result.conversationId.slice(0, 20)}...)`);
    console.log(`    Annotated: ${result.annotatedMessages}, Passed: ${result.passed}, Failed: ${result.failed}`);

    totalPassed += result.passed;
    totalFailed += result.failed;
  }

  console.log();
  console.log(`${colors.bold}Total: ${totalPassed} passed, ${totalFailed} failed${colors.reset}`);

  if (totalFailed > 0) {
    console.log(`\n${colors.red}${colors.bold}Some tests failed!${colors.reset}`);

    // Show failures
    for (const result of results) {
      const failures = result.details.filter(d => !d.passed);
      if (failures.length > 0) {
        console.log(`\n${colors.yellow}Failures in ${result.scenario}:${colors.reset}`);
        for (const f of failures) {
          console.log(`  [${f.messageIndex}] "${f.content}"`);
          console.log(`       Expected branch=${f.expectedBranch}, got ${f.actualAction}`);
          console.log(`       Similarity: ${f.similarity.toFixed(3)}`);
          if (f.notes) console.log(`       Notes: ${f.notes}`);
        }
      }
    }

    process.exit(1);
  } else {
    console.log(`\n${colors.green}${colors.bold}All tests passed!${colors.reset}`);
    process.exit(0);
  }
}

main().catch(console.error);
