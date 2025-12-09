import { Type } from '@sinclair/typebox';
import type { FastifyPluginAsyncTypebox } from '@fastify/type-provider-typebox';
import { contextService } from '@services/context';
import { factsService } from '@services/facts';

/**
 * Branches Routes
 * 
 * GET  /branches/:id         - Get branch details
 * GET  /branches/:id/context - Get branch context (messages + ancestor facts)
 * GET  /branches/:id/facts   - Get facts for branch
 * POST /branches/:id/facts   - Extract facts from branch
 */
const branchesRoutes: FastifyPluginAsyncTypebox = async (fastify) => {
  
  // Get branch details
  fastify.get(
    '/:branchId',
    {
      schema: {
        description: 'Get branch details including stats',
        tags: ['Branches'],
        params: Type.Object({
          branchId: Type.String(),
        }),
        response: {
          200: Type.Object({
            success: Type.Literal(true),
            data: Type.Object({
              id: Type.String(),
              conversationId: Type.String(),
              topic: Type.String(),
              parentId: Type.Optional(Type.String()),
              driftType: Type.Optional(Type.String()),
              depth: Type.Number(),
              messageCount: Type.Number(),
              factCount: Type.Number(),
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
      const { branchId } = request.params;

      const branch = await fastify.prisma.branch.findUnique({
        where: { id: branchId },
        include: {
          _count: { select: { messages: true, facts: true } },
        },
      });

      if (!branch) {
        return reply.status(404).send({
          success: false,
          error: { message: 'Branch not found' },
        });
      }

      return reply.send({
        success: true,
        data: {
          id: branch.id,
          conversationId: branch.conversationId,
          topic: branch.summary ?? 'Unknown',
          parentId: branch.parentId ?? undefined,
          driftType: branch.driftType ?? undefined,
          depth: branch.branchDepth,
          messageCount: branch._count.messages,
          factCount: branch._count.facts,
          createdAt: branch.createdAt.toISOString(),
          updatedAt: branch.updatedAt.toISOString(),
        },
      });
    }
  );

  // Get branch context (messages + ancestor facts)
  fastify.get(
    '/:branchId/context',
    {
      schema: {
        description: 'Get context for a branch (messages + ancestor facts)',
        tags: ['Branches'],
        params: Type.Object({
          branchId: Type.String(),
        }),
        querystring: Type.Object({
          maxMessages: Type.Optional(Type.Number({ default: 50 })),
          includeAncestorFacts: Type.Optional(Type.Boolean({ default: true })),
          maxAncestorDepth: Type.Optional(Type.Number({ default: 5 })),
        }),
        response: {
          200: Type.Object({
            success: Type.Literal(true),
            data: Type.Object({
              branchId: Type.String(),
              branchTopic: Type.String(),
              messages: Type.Array(
                Type.Object({
                  id: Type.String(),
                  role: Type.String(),
                  content: Type.String(),
                  createdAt: Type.String(),
                })
              ),
              facts: Type.Array(
                Type.Object({
                  branchId: Type.String(),
                  branchTopic: Type.String(),
                  isCurrent: Type.Boolean(),
                  items: Type.Array(
                    Type.Object({
                      id: Type.String(),
                      key: Type.String(),
                      value: Type.String(),
                      confidence: Type.Number(),
                    })
                  ),
                })
              ),
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
      const { branchId } = request.params;
      const { maxMessages, includeAncestorFacts, maxAncestorDepth } = request.query;

      const result = await contextService.get(branchId, {
        policy: { maxMessages, includeAncestorFacts, maxAncestorDepth },
      });

      if (!result.success) {
        return reply.status(400).send({
          success: false,
          error: { message: result.error?.message || 'Failed to get context' },
        });
      }

      const data = result.data!;
      return reply.send({
        success: true,
        data: {
          branchId: data.branchId,
          branchTopic: data.branchTopic,
          messages: data.messages,
          facts: data.allFacts.map((f) => ({
            branchId: f.branchId,
            branchTopic: f.branchTopic,
            isCurrent: f.isCurrent,
            items: f.facts,
          })),
        },
      });
    }
  );

  // Get facts for branch (with provenance)
  fastify.get(
    '/:branchId/facts',
    {
      schema: {
        description: 'Get existing facts for a branch with provenance',
        tags: ['Branches'],
        params: Type.Object({
          branchId: Type.String(),
        }),
        response: {
          200: Type.Object({
            success: Type.Literal(true),
            data: Type.Array(
              Type.Object({
                id: Type.String(),
                key: Type.String(),
                value: Type.String(),
                confidence: Type.Number(),
                messageIds: Type.Array(Type.String()),
                createdAt: Type.String(),
              })
            ),
          }),
        },
      },
    },
    async (request, reply) => {
      const { branchId } = request.params;

      const facts = await fastify.prisma.fact.findMany({
        where: { branchId },
        orderBy: { createdAt: 'asc' },
      });

      return reply.send({
        success: true,
        data: facts.map((f) => ({
          id: f.id,
          key: f.key,
          value: f.value,
          confidence: f.confidence,
          messageIds: f.messageIds,
          createdAt: f.createdAt.toISOString(),
        })),
      });
    }
  );

  // Extract facts from branch
  fastify.post(
    '/:branchId/facts',
    {
      schema: {
        description: 'Extract facts from a branch using LLM with provenance tracking',
        tags: ['Branches'],
        params: Type.Object({
          branchId: Type.String(),
        }),
        response: {
          200: Type.Object({
            success: Type.Literal(true),
            data: Type.Object({
              branchId: Type.String(),
              extractedCount: Type.Number(),
              facts: Type.Array(
                Type.Object({
                  id: Type.String(),
                  key: Type.String(),
                  value: Type.String(),
                  confidence: Type.Number(),
                  messageIds: Type.Array(Type.String()),
                  createdAt: Type.String(),
                })
              ),
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
      const { branchId } = request.params;

      const result = await factsService.extract(branchId);

      if (!result.success) {
        return reply.status(400).send({
          success: false,
          error: { message: result.error?.message || 'Extraction failed' },
        });
      }

      const data = result.data!;
      return reply.send({
        success: true,
        data: {
          branchId: data.branchId,
          extractedCount: data.extractedCount,
          facts: data.facts.map((f) => ({
            id: f.id,
            key: f.key,
            value: f.value,
            confidence: f.confidence,
            messageIds: f.messageIds,
            createdAt: f.createdAt.toISOString(),
          })),
        },
      });
    }
  );
};

export default branchesRoutes;
