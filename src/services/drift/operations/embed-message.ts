import type { DriftContext } from '../types';
import { embed } from '@services/local-embeddings';

/**
 * EmbedMessage Operation
 *
 * Generates embedding for the message content using the Python embedding server.
 * Preprocessing is enabled by default to strip conversational noise for better drift detection.
 */
export async function embedMessage(ctx: DriftContext): Promise<DriftContext> {
  // Preprocess + embed for better semantic separation
  ctx.embedding = await embed(ctx.content, true);
  ctx.reasonCodes.push('message_embedded');

  return ctx;
}
