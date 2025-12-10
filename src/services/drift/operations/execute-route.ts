import { factsService } from '@/services/facts';
import type { DriftContext } from '../types';
import { prisma } from '@plugins/prisma';
import { createLogger } from '@utils/logger';

const logger = createLogger('drift');

/**
 * ExecuteRoute Operation
 *
 * Creates message, creates branch if needed, updates centroid.
 */
export async function executeRoute(ctx: DriftContext): Promise<DriftContext> {
  if (!ctx.classification) {
    throw new Error('No classification result');
  }

  const { action, targetBranchId } = ctx.classification;

  let branchId: string;

  switch (action) {
    case 'STAY':
      // Use current branch
      if (!ctx.currentBranch) {
        throw new Error('No current branch to stay in');
      }
      branchId = ctx.currentBranch.id;
      break;

    case 'ROUTE':
      // Route to existing branch
      if (!targetBranchId) {
        throw new Error('ROUTE action requires targetBranchId');
      }
      branchId = targetBranchId;
      break;

    case 'BRANCH':
      // Create new branch
      const newBranch = await prisma.branch.create({
        data: {
          conversationId: ctx.conversationId,
          parentId: ctx.currentBranch?.id ?? null,
          summary: ctx.classification?.newBranchTopic ?? ctx.content.slice(0, 100),
          centroid: ctx.embedding ?? [],
          driftType: ctx.classification?.driftAction === 'BRANCH_NEW_CLUSTER' ? 'semantic' : 'functional',
        },
      });
      branchId = newBranch.id;
      ctx.reasonCodes.push('branch_created');
      break;
  }

  // Async fact extraction when leaving a branch (if enabled)
  if (ctx.extractFacts && action === 'BRANCH' && ctx.currentBranch) {
    // Fire and forget - don't block response
    factsService
      .extract(ctx.currentBranch.id)
      .catch((err) => logger.warn({ err, branchId: ctx.currentBranch?.id }, 'Async fact extraction failed'));
    ctx.reasonCodes.push('facts_extraction_triggered');
  }

  if (ctx.extractFacts && action === 'ROUTE' && ctx.currentBranch) {
    factsService
      .extract(ctx.currentBranch.id)
      .catch((err) => logger.warn({ err, branchId: ctx.currentBranch?.id }, 'Async fact extraction failed'));
    ctx.reasonCodes.push('facts_extraction_triggered');
  }

  // Create message
  const message = await prisma.message.create({
    data: {
      conversationId: ctx.conversationId,
      branchId,
      role: ctx.role,
      content: ctx.content,
      embedding: ctx.embedding ?? [],
      preprocessedEmbedding: [], // TODO: Store preprocessed embedding separately
    },
  });

  // Update branch centroid (weighted by role - user messages matter more)
  if (ctx.embedding && action !== 'BRANCH') {
    await updateCentroid(branchId, ctx.embedding, ctx.role);
  }

  // Load the branch for result
  const branch = await prisma.branch.findUniqueOrThrow({
    where: { id: branchId },
  });

  ctx.message = message;
  ctx.branch = branch;
  ctx.reasonCodes.push('message_created');

  return ctx;
}

/**
 * User message weight multiplier.
 * User messages define the topic, assistant responses elaborate.
 * Weight user messages 3x more heavily in centroid calculation.
 */
const USER_WEIGHT = 3.0;
const ASSISTANT_WEIGHT = 1.0;

async function updateCentroid(
  branchId: string,
  newEmbedding: number[],
  role: 'user' | 'assistant' = 'user'
): Promise<void> {
  const branch = await prisma.branch.findUniqueOrThrow({
    where: { id: branchId },
    include: { _count: { select: { messages: true } } },
  });

  const oldCentroid = branch.centroid as number[];

  if (oldCentroid.length === 0) {
    await prisma.branch.update({
      where: { id: branchId },
      data: { centroid: newEmbedding },
    });
    return;
  }

  // Weighted running average
  // User messages get 3x weight, assistant gets 1x
  const weight = role === 'user' ? USER_WEIGHT : ASSISTANT_WEIGHT;
  const messageCount = branch._count.messages;
  
  // Effective count treats user messages as worth more
  // new = old + weight * (new - old) / (n + weight - 1)
  const effectiveDivisor = messageCount + weight - 1;
  
  const updatedCentroid = oldCentroid.map(
    (val, i) => val + (weight * ((newEmbedding[i] ?? 0) - val)) / effectiveDivisor
  );

  await prisma.branch.update({
    where: { id: branchId },
    data: { centroid: updatedCentroid },
  });
}
