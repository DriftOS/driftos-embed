import type { FactsContext, ExtractedFact } from '../types';
import { getConfig } from '@plugins/env';
import { getModelConfig, getResponseFormat, getApiKey } from '@config/llm-models';
import { createLogger } from '@utils/logger';

const logger = createLogger('facts');

// JSON Schema for models that support it (Scout, Maverick, GPT-4o)
const FACT_SCHEMA = {
  name: 'facts_and_topic',
  schema: {
    type: 'object',
    properties: {
      branchTopic: {
        type: 'string',
        description: 'A concise 3-6 word topic name for this conversation branch',
      },
      facts: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            key: { type: 'string' },
            value: { type: 'string' },
            confidence: { type: 'number' },
            messageIds: {
              type: 'array',
              items: { type: 'string' },
            },
          },
          required: ['key', 'value', 'confidence', 'messageIds'],
        },
      },
    },
    required: ['branchTopic', 'facts'],
  },
};

const EXTRACTION_PROMPT = `Analyze this conversation branch and extract a topic name and key facts.
Return a JSON object with "branchTopic" and "facts".

RULES:
1. branchTopic: A concise 3-6 word topic name (e.g., "Buying a house in London", "Planning dinner options", "Car repair advice")
2. ONE fact per concept - consolidate multiple mentions into a single fact
3. messageIds: Include ALL message IDs that mention/reinforce this fact (provenance)
4. Use the LATEST/MOST SPECIFIC value if something evolves
5. Use snake_case keys (e.g., "destination", "budget_range", "hotel_preference")
6. Confidence scoring:
   - 1.0 = explicitly stated in multiple messages (reinforced)
   - 0.9 = explicitly stated once
   - 0.7 = clearly implied
   - 0.5 = inferred

OUTPUT FORMAT:
{
  "branchTopic": "3-6 word topic name",
  "facts": [
    {"key": "destination", "value": "Paris", "confidence": 1.0, "messageIds": ["abc123", "def456"]}
  ]
}

EXTRACT:
- Decisions made
- Preferences stated
- Key entities (places, dates, people, amounts)
- Constraints or requirements

DO NOT extract:
- Questions without answers
- Trivial conversational elements
- Information NOT explicitly mentioned in the messages
- Inferred or assumed facts that aren't directly stated

IMPORTANT: Only extract facts that are DIRECTLY STATED in the conversation. Do not hallucinate or infer facts that are not explicitly mentioned. If unsure, do not include the fact.`;

async function callOpenAICompatible(
  prompt: string,
  apiKey: string,
  modelConfig: ReturnType<typeof getModelConfig>
): Promise<string> {
  const responseFormat = getResponseFormat(modelConfig.id, FACT_SCHEMA);
  
  // OpenAI uses max_completion_tokens, Groq uses max_tokens
  const maxTokensKey = modelConfig.provider === 'openai' ? 'max_completion_tokens' : 'max_tokens';
  
  // GPT-5 family doesn't support temperature
  const supportsTemp = modelConfig.supportsTemperature !== false;

  const response = await fetch(`${modelConfig.baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: modelConfig.id,
      messages: [{ role: 'user', content: prompt }],
      ...(supportsTemp && { temperature: modelConfig.defaultTemperature }),
      [maxTokensKey]: modelConfig.maxTokens,
      ...(responseFormat && { response_format: responseFormat }),
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`${modelConfig.provider} call failed: ${response.status} - ${error}`);
  }

  const data = await response.json();
  return data.choices[0].message.content;
}

async function callAnthropic(
  prompt: string,
  apiKey: string,
  modelConfig: ReturnType<typeof getModelConfig>
): Promise<string> {
  const response = await fetch(`${modelConfig.baseUrl}/messages`, {
    method: 'POST',
    headers: {
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: modelConfig.id,
      max_tokens: modelConfig.maxTokens,
      messages: [{ role: 'user', content: prompt + '\n\nRespond with ONLY valid JSON, no markdown or explanation.' }],
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Anthropic call failed: ${response.status} - ${error}`);
  }

  const data = await response.json();
  return data.content[0].text;
}

export async function extractFacts(ctx: FactsContext): Promise<FactsContext> {
  if (!ctx.messages || ctx.messages.length === 0) {
    ctx.extractedFacts = [];
    return ctx;
  }

  const config = getConfig();
  const modelConfig = getModelConfig(config.LLM_MODEL);
  const apiKey = getApiKey(modelConfig.provider, config);

  const conversationText = ctx.messages
    .map((m) => `[${m.id}] [${m.role.toUpperCase()}]: ${m.content}`)
    .join('\n\n');

  const prompt = `${EXTRACTION_PROMPT}

Conversation:
${conversationText}

Output JSON with consolidated facts and full provenance (all supporting messageIds).`;

  // Call the appropriate provider
  let content: string;
  try {
    if (modelConfig.provider === 'anthropic') {
      content = await callAnthropic(prompt, apiKey, modelConfig);
    } else {
      content = await callOpenAICompatible(prompt, apiKey, modelConfig);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // Clean log for rate limits, full error for other failures
    if (message.includes('429') || message.includes('rate_limit')) {
      logger.warn({ branchId: ctx.branchId }, 'Facts extraction rate limited, skipping');
    } else {
      logger.warn({ err: message, branchId: ctx.branchId }, 'LLM call failed');
    }
    ctx.extractedFacts = [];
    ctx.reasonCodes.push('llm_call_failed');
    return ctx;
  }

  // Parse response - handle potential markdown wrapping and empty responses
  if (!content || content.trim() === '') {
    logger.warn('Empty response from LLM');
    ctx.extractedFacts = [];
    ctx.reasonCodes.push('empty_llm_response');
    return ctx;
  }

  let jsonStr = content.trim();
  if (jsonStr.startsWith('```')) {
    jsonStr = jsonStr.replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '');
  }

  let parsed: { branchTopic?: string; facts?: unknown[] };
  try {
    parsed = JSON.parse(jsonStr);
  } catch {
    logger.warn({ raw: content.slice(0, 200) }, 'JSON parse failed');
    ctx.extractedFacts = [];
    ctx.reasonCodes.push('json_parse_failed');
    return ctx;
  }

  const facts: ExtractedFact[] = (parsed.facts ?? []).map((f: any) => ({
    key: String(f.key),
    value: String(f.value),
    confidence: Number(f.confidence),
    messageIds: Array.isArray(f.messageIds) ? f.messageIds : [f.messageIds].filter(Boolean),
  }));

  ctx.extractedFacts = facts.filter((f) => f.confidence >= ctx.policy.minConfidence);
  ctx.branchTopic = parsed.branchTopic;
  ctx.reasonCodes.push(`extracted_${ctx.extractedFacts.length}_facts`);

  if (ctx.branchTopic) {
    ctx.reasonCodes.push('topic_extracted');
  }

  return ctx;
}
