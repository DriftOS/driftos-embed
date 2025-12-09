import { Type } from '@sinclair/typebox';
import type { FastifyPluginAsyncTypebox } from '@fastify/type-provider-typebox';

/**
 * Conversations Routes
 * 
 * GET  /conversations/:id          - Get conversation summary
 * GET  /conversations/:id/branches - List all branches
 * GET  /conversations/:id/context  - Get full conversation context (all branches)
 */
const conversationsRoutes: FastifyPluginAsyncTypebox = async (fastify) => {
  
  // Get conversation summary
  fastify.get(
    '/:conversationId',
    {
      schema: {
        description: 'Get conversation summary with branch count and message stats',
        tags: ['Conversations'],
        params: Type.Object({
          conversationId: Type.String(),
        }),
        response: {
          200: Type.Object({
            success: Type.Literal(true),
            data: Type.Object({
              id: Type.String(),
              branchCount: Type.Number(),
              messageCount: Type.Number(),
              factCount: Type.Number(),
              currentBranchId: Type.Optional(Type.String()),
              createdAt: Type.String(),
              updatedAt: Type.String(),
            }),
          }),
          404: Type.Object({
            success: Type.Literal(false),
            error: Type.Object({ message: Type.String() }),
          }),
        },
      },
    },
    async (request, reply) => {
      const { conversationId } = request.params;

      const conversation = await fastify.prisma.conversation.findUnique({
        where: { id: conversationId },
        include: {
          _count: { select: { branches: true, messages: true } },
          branches: {
            orderBy: { updatedAt: 'desc' },
            take: 1,
            select: { id: true },
          },
        },
      });

      if (!conversation) {
        return reply.status(404).send({
          success: false,
          error: { message: 'Conversation not found' },
        });
      }

      // Count facts across all branches
      const factCount = await fastify.prisma.fact.count({
        where: { branch: { conversationId } },
      });

      return reply.send({
        success: true,
        data: {
          id: conversation.id,
          branchCount: conversation._count.branches,
          messageCount: conversation._count.messages,
          factCount,
          currentBranchId: conversation.branches[0]?.id,
          createdAt: conversation.createdAt.toISOString(),
          updatedAt: conversation.updatedAt.toISOString(),
        },
      });
    }
  );

  // List branches for a conversation
  fastify.get(
    '/:conversationId/branches',
    {
      schema: {
        description: 'List all branches for a conversation',
        tags: ['Conversations'],
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
                depth: Type.Number(),
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
          _count: { select: { messages: true, facts: true } },
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
          depth: b.branchDepth,
          createdAt: b.createdAt.toISOString(),
          updatedAt: b.updatedAt.toISOString(),
        })),
      });
    }
  );

  // Get full conversation context (messages from all branches + facts)
  fastify.get(
    '/:conversationId/context',
    {
      schema: {
        description: 'Get full conversation context - messages from current branch lineage + all facts',
        tags: ['Conversations'],
        params: Type.Object({
          conversationId: Type.String(),
        }),
        querystring: Type.Object({
          branchId: Type.Optional(Type.String({ description: 'Branch to get context for (defaults to most recent)' })),
          maxMessages: Type.Optional(Type.Number({ default: 50 })),
          includeFacts: Type.Optional(Type.Boolean({ default: true })),
        }),
        response: {
          200: Type.Object({
            success: Type.Literal(true),
            data: Type.Object({
              conversationId: Type.String(),
              currentBranchId: Type.String(),
              branches: Type.Array(
                Type.Object({
                  id: Type.String(),
                  topic: Type.String(),
                  depth: Type.Number(),
                  messageCount: Type.Number(),
                })
              ),
              messages: Type.Array(
                Type.Object({
                  id: Type.String(),
                  branchId: Type.String(),
                  role: Type.String(),
                  content: Type.String(),
                  createdAt: Type.String(),
                })
              ),
              facts: Type.Array(
                Type.Object({
                  id: Type.String(),
                  branchId: Type.String(),
                  branchTopic: Type.String(),
                  key: Type.String(),
                  value: Type.String(),
                  confidence: Type.Number(),
                })
              ),
              stats: Type.Object({
                totalMessages: Type.Number(),
                totalFacts: Type.Number(),
                branchDepth: Type.Number(),
              }),
            }),
          }),
          404: Type.Object({
            success: Type.Literal(false),
            error: Type.Object({ message: Type.String() }),
          }),
        },
      },
    },
    async (request, reply) => {
      const { conversationId } = request.params;
      const { branchId, maxMessages = 50, includeFacts = true } = request.query;

      // Get conversation
      const conversation = await fastify.prisma.conversation.findUnique({
        where: { id: conversationId },
      });

      if (!conversation) {
        return reply.status(404).send({
          success: false,
          error: { message: 'Conversation not found' },
        });
      }

      // Get target branch (specified or most recent)
      let targetBranch;
      if (branchId) {
        targetBranch = await fastify.prisma.branch.findUnique({
          where: { id: branchId },
        });
      } else {
        targetBranch = await fastify.prisma.branch.findFirst({
          where: { conversationId },
          orderBy: { updatedAt: 'desc' },
        });
      }

      if (!targetBranch) {
        return reply.status(404).send({
          success: false,
          error: { message: 'No branches found' },
        });
      }

      // Walk up the branch tree to get lineage
      const branchLineage: string[] = [targetBranch.id];
      let currentBranch = targetBranch;
      while (currentBranch.parentId) {
        branchLineage.unshift(currentBranch.parentId);
        const parent = await fastify.prisma.branch.findUnique({
          where: { id: currentBranch.parentId },
        });
        if (!parent) break;
        currentBranch = parent;
      }

      // Get branches in lineage
      const branches = await fastify.prisma.branch.findMany({
        where: { id: { in: branchLineage } },
        include: { _count: { select: { messages: true } } },
        orderBy: { branchDepth: 'asc' },
      });

      // Get messages from lineage branches
      const messages = await fastify.prisma.message.findMany({
        where: { branchId: { in: branchLineage } },
        orderBy: { createdAt: 'asc' },
        take: maxMessages,
        select: {
          id: true,
          branchId: true,
          role: true,
          content: true,
          createdAt: true,
        },
      });

      // Get facts if requested
      let facts: Array<{
        id: string;
        branchId: string;
        branchTopic: string;
        key: string;
        value: string;
        confidence: number;
      }> = [];

      if (includeFacts) {
        const rawFacts = await fastify.prisma.fact.findMany({
          where: { branchId: { in: branchLineage } },
          include: { branch: { select: { summary: true } } },
          orderBy: { createdAt: 'asc' },
        });

        facts = rawFacts.map((f) => ({
          id: f.id,
          branchId: f.branchId,
          branchTopic: f.branch.summary ?? 'Unknown',
          key: f.key,
          value: f.value,
          confidence: f.confidence,
        }));
      }

      return reply.send({
        success: true,
        data: {
          conversationId,
          currentBranchId: targetBranch.id,
          branches: branches.map((b) => ({
            id: b.id,
            topic: b.summary ?? 'Unknown',
            depth: b.branchDepth,
            messageCount: b._count.messages,
          })),
          messages: messages.map((m) => ({
            ...m,
            createdAt: m.createdAt.toISOString(),
          })),
          facts,
          stats: {
            totalMessages: messages.length,
            totalFacts: facts.length,
            branchDepth: targetBranch.branchDepth,
          },
        },
      });
    }
  );
};

export default conversationsRoutes;
