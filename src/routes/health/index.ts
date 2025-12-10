import { Type } from '@sinclair/typebox';
import type { FastifyPluginAsyncTypebox } from '@fastify/type-provider-typebox';
import { healthCheck as embeddingHealthCheck } from '@services/local-embeddings';

const healthRoutes: FastifyPluginAsyncTypebox = async (fastify) => {
  // Health check endpoint
  fastify.get(
    '/health',
    {
      schema: {
        description: 'Health check endpoint',
        tags: ['Health'],
        response: {
          200: Type.Object({
            status: Type.Literal('ok'),
            timestamp: Type.String(),
            uptime: Type.Number(),
            environment: Type.String(),
          }),
        },
      },
    },
    async (_request, reply) => {
      return reply.send({
        status: 'ok',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        environment: process.env.NODE_ENV || 'development',
      });
    }
  );

  // Readiness check endpoint
  fastify.get(
    '/ready',
    {
      schema: {
        description: 'Readiness check endpoint - checks database connection',
        tags: ['Health'],
        response: {
          200: Type.Object({
            status: Type.Literal('ready'),
            services: Type.Object({
              database: Type.Boolean(),
              embeddingServer: Type.Boolean(),
            }),
            embeddingServerUrl: Type.Optional(Type.String()),
            timestamp: Type.String(),
          }),
          503: Type.Object({
            status: Type.Literal('not_ready'),
            services: Type.Object({
              database: Type.Boolean(),
              embeddingServer: Type.Boolean(),
            }),
            embeddingServerUrl: Type.Optional(Type.String()),
            embeddingError: Type.Optional(Type.String()),
            timestamp: Type.String(),
          }),
        },
      },
    },
    async (_request, reply) => {
      let isDatabaseReady = false;
      let isEmbeddingServerReady = false;
      let embeddingError: string | undefined;
      const embeddingServerUrl = process.env.EMBEDDING_SERVER_URL ?? 'http://localhost:8100';

      try {
        // Check database connection
        await fastify.prisma.$queryRaw`SELECT 1`;
        isDatabaseReady = true;
      } catch (err) {
        fastify.log.error({ err }, 'Database health check failed');
      }

      try {
        // Check embedding server connection
        await embeddingHealthCheck();
        isEmbeddingServerReady = true;
      } catch (err) {
        embeddingError = err instanceof Error ? err.message : String(err);
        fastify.log.error({ err, embeddingServerUrl }, 'Embedding server health check failed');
      }

      const allReady = isDatabaseReady && isEmbeddingServerReady;
      const status = allReady ? 'ready' : 'not_ready';
      const statusCode = allReady ? 200 : 503;

      return reply.status(statusCode).send({
        status,
        services: {
          database: isDatabaseReady,
          embeddingServer: isEmbeddingServerReady,
        },
        embeddingServerUrl,
        embeddingError,
        timestamp: new Date().toISOString(),
      });
    }
  );
};

export default healthRoutes;
