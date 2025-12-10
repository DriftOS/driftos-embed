import { Type } from '@sinclair/typebox';
import type { FastifyPluginAsyncTypebox } from '@fastify/type-provider-typebox';
import { driftService } from '@services/drift';

const driftRoutes: FastifyPluginAsyncTypebox = async (fastify) => {
  // Route a message
  fastify.post(
    '/route',
    {
      schema: {
        description: 'Route a message to the appropriate branch using drift detection',
        tags: ['Drift'],
        body: Type.Object({
          conversationId: Type.String(),
          content: Type.String(),
          role: Type.Optional(Type.Union([Type.Literal('user'), Type.Literal('assistant')])),
          currentBranchId: Type.Optional(Type.String()),
          extractFacts: Type.Optional(Type.Boolean()),
        }),
        response: {
          200: Type.Object({
            success: Type.Literal(true),
            data: Type.Object({
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
              branchId: Type.String(),
              messageId: Type.String(),
              previousBranchId: Type.Optional(Type.String()),
              isNewBranch: Type.Boolean(),
              isNewCluster: Type.Boolean(),
              reason: Type.String(),
              branchTopic: Type.Optional(Type.String()),
              clusterId: Type.Optional(Type.String()),
              confidence: Type.Number(),
              similarity: Type.Number(),
              reasonCodes: Type.Array(Type.String()),
            }),
          }),
          400: Type.Object({
            success: Type.Literal(false),
            error: Type.Object({
              message: Type.String(),
            }),
          }),
        },
      },
    },
    async (request, reply) => {
      const { conversationId, content, role, currentBranchId, extractFacts } = request.body;

      const result = await driftService.route(conversationId, content, {
        role,
        currentBranchId,
        extractFacts,
      });

      if (!result.success) {
        return reply.status(400).send({
          success: false,
          error: { message: result.error?.message || 'Routing failed' },
        });
      }

      return reply.send({
        success: true,
        data: result.data!,
      });
    }
  );
  
  // List branches for a conversation
  fastify.get(
    '/branches/:conversationId',
    {
      schema: {
        description: 'List all branches for a conversation',
        tags: ['Drift'],
        params: Type.Object({
          conversationId: Type.String(),
        }),
        response: {
          200: Type.Object({
            success: Type.Literal(true),
            data: Type.Array(
              Type.Object({
                id: Type.String(),
                topic: Type.String(),
                messageCount: Type.Number(),
                factCount: Type.Number(),
                parentId: Type.Optional(Type.String()),
                driftType: Type.Optional(Type.String()),
                createdAt: Type.String(),
                updatedAt: Type.String(),
              })
            ),
          }),
        },
      },
    },
    async (request, reply) => {
      const { conversationId } = request.params;

      const branches = await fastify.prisma.branch.findMany({
        where: { conversationId },
        include: {
          _count: {
            select: { messages: true, facts: true },
          },
        },
        orderBy: { createdAt: 'asc' },
      });

      return reply.send({
        success: true,
        data: branches.map((b) => ({
          id: b.id,
          topic: b.summary ?? 'Unknown',
          messageCount: b._count.messages,
          factCount: b._count.facts,
          parentId: b.parentId ?? undefined,
          driftType: b.driftType ?? undefined,
          createdAt: b.createdAt.toISOString(),
          updatedAt: b.updatedAt.toISOString(),
        })),
      });
    }
  );

  // Health check
  fastify.get(
    '/health',
    {
      schema: {
        description: 'Check if Drift service is healthy',
        tags: ['Drift'],
        response: {
          200: Type.Object({
            status: Type.String(),
            service: Type.String(),
          }),
        },
      },
    },
    async (_request, reply) => {
      const health = await driftService.healthCheck();
      return reply.send(health);
    }
  );
};

export default driftRoutes;
