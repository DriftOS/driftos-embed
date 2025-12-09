import type { RouteAction, Classification } from '@/services/drift/types';

export function parseResponse(response: string): Classification {
  const parsed = JSON.parse(response);

  return {
    action: parsed.action as RouteAction,
    driftAction: parsed.driftAction ?? 'STAY',
    targetBranchId: parsed.targetBranchId || undefined,
    newBranchTopic: parsed.newBranchTopic || undefined,
    reason: parsed.reason || 'Unknown',
    confidence: parsed.confidence || 0.5,
    similarity: parsed.similarity || 0,
  };
}
