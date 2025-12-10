import { Type } from '@sinclair/typebox';
import type { FastifyPluginAsyncTypebox } from '@fastify/type-provider-typebox';

/**
 * Conversations Routes
 * 
 * GET  /conversations              - List conversations by prefix
 * GET  /conversations/:id          - Get conversation summary
 * GET  /conversations/:id/branches - List all branches
 * GET  /conversations/:id/context  - Get full conversation context (all branches)
 */
const conversationsRoutes: FastifyPluginAsyncTypebox = async (fastify) => {
  
  // List conversations by prefix (for device-based filtering)
  fastify.get(
    '/',
    {
      schema: {
        description: 'List conversations filtered by ID prefix (for device-based access)',
        tags: ['Conversations'],
        querystring: Type.Object({
          prefix: Type.String({ description: 'Conversation ID prefix to filter by' }),
          limit: Type.Optional(Type.Number({ default: 50, maximum: 100 })),
        }),
        response: {
          200: Type.Object({
            success: Type.Literal(true),
            data: Type.Array(
              Type.Object({
                id: Type.String(),
                title: Type.String(),
                messageCount: Type.Number(),
                branchCount: Type.Number(),
                createdAt: Type.String(),
                updatedAt: Type.String(),
              })
            ),
          }),
        },
      },
    },
    async (request, reply) => {
      const { prefix, limit = 50 } = request.query;

      // Find conversations with matching prefix
      const conversations = await fastify.prisma.conversation.findMany({
        where: {
          id: { startsWith: prefix },
        },
        include: {
          _count: { select: { branches: true, messages: true } },
          messages: {
            take: 1,
            orderBy: { createdAt: 'asc' },
            select: { content: true },
          },
        },
        orderBy: { updatedAt: 'desc' },
        take: limit,
      });

      return reply.send({
        success: true,
        data: conversations.map((c) => ({
          id: c.id,
          title: c.messages[0]?.content?.slice(0, 50) || 'New Chat',
          messageCount: c._count.messages,
          branchCount: c._count.branches,
          createdAt: c.createdAt.toISOString(),
          updatedAt: c.updatedAt.toISOString(),
        })),
      });
    }
  );

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
          allBranches: Type.Optional(Type.Boolean({ default: false, description: 'If true, return messages from ALL branches, not just lineage' })),
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
                  driftAction: Type.Optional(Type.String()),
                  driftReason: Type.Optional(Type.String()),
                  driftMetadata: Type.Optional(Type.Any()),
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
      const { branchId, maxMessages = 50, includeFacts = true, allBranches = false } = request.query;

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

      // Determine which branches to include
      let branchIds: string[];
      
      if (allBranches) {
        // Get ALL branches for this conversation
        const allBranchRecords = await fastify.prisma.branch.findMany({
          where: { conversationId },
          select: { id: true },
        });
        branchIds = allBranchRecords.map(b => b.id);
      } else {
        // Walk up the branch tree to get lineage only
        branchIds = [targetBranch.id];
        let currentBranch = targetBranch;
        while (currentBranch.parentId) {
          branchIds.unshift(currentBranch.parentId);
          const parent = await fastify.prisma.branch.findUnique({
            where: { id: currentBranch.parentId },
          });
          if (!parent) break;
          currentBranch = parent;
        }
      }

      // Get branches
      const branches = await fastify.prisma.branch.findMany({
        where: { id: { in: branchIds } },
        include: { _count: { select: { messages: true } } },
        orderBy: { branchDepth: 'asc' },
      });

      // Get messages from selected branches
      const messages = await fastify.prisma.message.findMany({
        where: { branchId: { in: branchIds } },
        orderBy: { createdAt: 'asc' },
        take: maxMessages,
        select: {
          id: true,
          branchId: true,
          role: true,
          content: true,
          driftAction: true,
          driftReason: true,
          driftMetadata: true,
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
          where: { branchId: { in: branchIds } },
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
