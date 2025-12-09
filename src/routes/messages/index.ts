import { Type } from '@sinclair/typebox';
import type { FastifyPluginAsyncTypebox } from '@fastify/type-provider-typebox';
import { driftService } from '@services/drift';

/**
 * Messages Routes
 * 
 * POST /messages - Send a message (routes to correct branch via drift detection)
 */
const messagesRoutes: FastifyPluginAsyncTypebox = async (fastify) => {
  fastify.post(
    '/',
    {
      schema: {
        description: 'Send a message - automatically routes to the correct branch using drift detection',
        tags: ['Messages'],
        body: Type.Object({
          conversationId: Type.String({ description: 'Conversation ID (created if not exists)' }),
          content: Type.String({ description: 'Message content' }),
          role: Type.Optional(Type.Union([Type.Literal('user'), Type.Literal('assistant')])),
          currentBranchId: Type.Optional(Type.String({ description: 'Current branch (auto-detected if omitted)' })),
        }),
        response: {
          200: Type.Object({
            success: Type.Literal(true),
            data: Type.Object({
              messageId: Type.String(),
              branchId: Type.String(),
              conversationId: Type.String(),
              action: Type.Union([
                Type.Literal('STAY'),
                Type.Literal('ROUTE'),
                Type.Literal('BRANCH'),
              ]),
              driftAction: Type.Union([
                Type.Literal('STAY'),
                Type.Literal('BRANCH_SAME_CLUSTER'),
                Type.Literal('BRANCH_NEW_CLUSTER'),
              ]),
              isNewBranch: Type.Boolean(),
              isNewCluster: Type.Boolean(),
              branchTopic: Type.Optional(Type.String()),
              similarity: Type.Number(),
              confidence: Type.Number(),
              reason: Type.String(),
            }),
          }),
          400: Type.Object({
            success: Type.Literal(false),
            error: Type.Object({ message: Type.String() }),
          }),
        },
      },
    },
    async (request, reply) => {
      const { conversationId, content, role, currentBranchId } = request.body;

      const result = await driftService.route(conversationId, content, {
        role,
        currentBranchId,
      });

      if (!result.success) {
        return reply.status(400).send({
          success: false,
          error: { message: result.error?.message || 'Failed to route message' },
        });
      }

      const data = result.data!;
      return reply.send({
        success: true,
        data: {
          messageId: data.messageId,
          branchId: data.branchId,
          conversationId,
          action: data.action,
          driftAction: data.driftAction,
          isNewBranch: data.isNewBranch,
          isNewCluster: data.isNewCluster,
          branchTopic: data.branchTopic,
          similarity: data.similarity,
          confidence: data.confidence,
          reason: data.reason,
        },
      });
    }
  );
};

export default messagesRoutes;
