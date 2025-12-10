import type { FactsContext } from '../types';
import { prisma } from '@plugins/prisma';
import { createLogger } from '@utils/logger';

const logger = createLogger('facts');

export async function saveFacts(ctx: FactsContext): Promise<FactsContext> {
  // Update branch topic if extracted
  if (ctx.branchTopic) {
    logger.debug({ branchId: ctx.branchId, branchTopic: ctx.branchTopic }, 'Updating branch summary');
    await prisma.branch.update({
      where: { id: ctx.branchId },
      data: { summary: ctx.branchTopic },
    });
    ctx.reasonCodes.push('branch_topic_updated');
  }

  if (!ctx.extractedFacts || ctx.extractedFacts.length === 0) {
    ctx.savedFacts = [];
    return ctx;
  }

  // Delete existing facts for this branch (replace strategy)
  await prisma.fact.deleteMany({
    where: { branchId: ctx.branchId },
  });

  // Create new facts with provenance
  const facts = await prisma.$transaction(
    ctx.extractedFacts.map((f) =>
      prisma.fact.create({
        data: {
          branchId: ctx.branchId,
          key: f.key,
          value: f.value,
          confidence: f.confidence,
          messageIds: f.messageIds,
        },
      })
    )
  );

  ctx.savedFacts = facts;
  ctx.reasonCodes.push(`facts_saved_${facts.length}`);

  return ctx;
}
