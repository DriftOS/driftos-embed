import type { DriftContext } from '../types';
import { embed } from '@services/local-embeddings';

/**
 * EmbedMessage Operation
 *
 * Generates embedding for the message content using the Python embedding server.
 * Preprocessing disabled - it hurts follow-up question detection by stripping contextual words.
 */
export async function embedMessage(ctx: DriftContext): Promise<DriftContext> {
  // No preprocessing - raw text preserves contextual relationships better
  ctx.embedding = await embed(ctx.content, false);
  ctx.reasonCodes.push('message_embedded');

  return ctx;
}
