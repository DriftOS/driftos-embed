.PHONY: help setup up down restart logs dev dev-api dev-embed test test-watch lint format typecheck build clean generate db-extensions db-migrate db-studio db-push db-reset docker-clean all test-drift test-embed

# Port configuration (match docker-compose.yml)
API_PORT := 3000
PROMETHEUS_PORT := 9091
GRAFANA_PORT := 3002
EMBEDDING_PORT := 8100

# Default target - show help
help:
	@echo "🚀 DriftOS Embed - Make Commands"
	@echo ""
	@echo "⚡ Quick Start:"
	@echo "  make up             - 🎯 Start everything (Docker + DB + migrations)"
	@echo "  make dev            - Start BOTH servers (embedding + API)"
	@echo "  make down           - Stop all services"
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  make setup          - Initial project setup (run once)"
	@echo "  make install        - Install dependencies only"
	@echo "  make embed-setup    - Setup Python embedding server"
	@echo ""
	@echo "🐳 Docker Services:"
	@echo "  make docker-up      - Start Docker services only"
	@echo "  make restart        - Restart all Docker services"
	@echo "  make logs           - View Docker logs"
	@echo ""
	@echo "🚀 Development:"
	@echo "  make dev            - Start BOTH servers (embedding + API)"
	@echo "  make dev-api        - Start Node API server only"
	@echo "  make dev-embed      - Start Python embedding server only"
	@echo "  make build          - Build for production"
	@echo ""
	@echo "🧪 Testing & Quality:"
	@echo "  make test           - Run all tests"
	@echo "  make test-drift     - Run drift detection test script"
	@echo "  make test-embed     - Test embedding server endpoints"
	@echo "  make test-watch     - Run tests in watch mode"
	@echo "  make lint           - Run ESLint"
	@echo "  make format         - Format code with Prettier"
	@echo "  make typecheck      - Run TypeScript type checking"
	@echo ""
	@echo "🗄️  Database:"
	@echo "  make db-extensions  - Install PostgreSQL extensions (pgvector, etc.)"
	@echo "  make db-migrate     - Run Prisma migrations"
	@echo "  make db-studio      - Open Prisma Studio"
	@echo "  make db-push        - Push schema changes"
	@echo "  make db-reset       - Reset database (WARNING: destructive)"
	@echo ""
	@echo "🎨 Generators:"
	@echo "  make generate       - Generate service (interactive)"
	@echo "  make dashboards     - Generate Grafana dashboards"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make docker-clean   - Remove Docker volumes (WARNING: destructive)"
	@echo ""
	@echo "⚡ Quick Combos:"
	@echo "  make all            - setup + up + db-migrate + dev"
	@echo ""

# Initial setup (run once)
setup: install embed-setup
	@echo "🚀 Running initial setup..."
	@./setup.sh

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	@npm install

# Setup Python embedding server
embed-setup:
	@echo "🐍 Setting up Python embedding server..."
	@cd embedding-server && \
		/opt/homebrew/bin/python3.12 -m venv .venv && \
		. .venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt

# Start Python embedding server only
dev-embed:
	@echo "🧠 Starting embedding server (paraphrase-MiniLM-L6-v2)..."
	@cd embedding-server && . .venv/bin/activate && uvicorn server:app --host 0.0.0.0 --port $(EMBEDDING_PORT) --reload

# Start Node API server only
dev-api:
	@echo "🚀 Starting API server on port $(API_PORT)..."
	@npm run dev

# Start BOTH servers (main dev command)
dev:
	@echo "🚀 Starting DriftOS development environment..."
	@echo ""
	@echo "📍 Services:"
	@echo "   • Embedding:   http://localhost:$(EMBEDDING_PORT) (paraphrase-MiniLM-L6-v2)"
	@echo "   • API:         http://localhost:$(API_PORT)"
	@echo "   • Swagger:     http://localhost:$(API_PORT)/documentation"
	@echo ""
	@cleanup() { pkill -9 -f "uvicorn server:app" 2>/dev/null; wait 2>/dev/null; }; \
	trap cleanup EXIT; \
	(cd embedding-server && . .venv/bin/activate && uvicorn server:app --host 0.0.0.0 --port $(EMBEDDING_PORT) --reload --log-level warning 2>/dev/null) & \
	sleep 3 && npm run dev; \
	echo "👋 Shutting down..."

# Docker commands
docker-up:
	@echo "🐳 Starting Docker services..."
	@npm run docker:up

down:
	@echo "🛑 Stopping Docker services..."
	@npm run docker:down
	@pkill -f "uvicorn server:app" 2>/dev/null || true

restart: down docker-up

logs:
	@echo "📜 Showing Docker logs..."
	@npm run docker:logs

# Initialize PostgreSQL extensions (required for pgvector, etc.)
db-extensions:
	@echo "🔌 Ensuring PostgreSQL extensions are installed..."
	@docker exec driftos_embed_postgres psql -U postgres -d driftos_embed -c 'CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; CREATE EXTENSION IF NOT EXISTS "pgcrypto"; CREATE EXTENSION IF NOT EXISTS "vector";' 2>/dev/null || true

# Main entry point
up: docker-up
	@echo "⏳ Waiting for PostgreSQL to be ready..."
	@sleep 8
	@$(MAKE) db-extensions
	@echo "🗄️  Pushing database schema..."
	@npm run db:push
	@echo "📊 Generating Grafana dashboards..."
	@npm run generate:dashboards || echo "⚠️  No orchestrators found yet"
	@echo ""
	@echo "✨ Everything is ready!"
	@echo ""
	@echo "📍 Services available:"
	@echo "   • API:         http://localhost:$(API_PORT)"
	@echo "   • Swagger:     http://localhost:$(API_PORT)/documentation"
	@echo "   • Prometheus:  http://localhost:$(PROMETHEUS_PORT)"
	@echo "   • Grafana:     http://localhost:$(GRAFANA_PORT) (admin/admin)"
	@echo "   • Embedding:   http://localhost:$(EMBEDDING_PORT)"
	@echo ""
	@echo "🚀 Run 'make dev' to start both servers!"
	@echo ""

# Open Grafana in browser
grafana:
	@echo "🎨 Opening Grafana..."
	@open http://localhost:$(GRAFANA_PORT) || xdg-open http://localhost:$(GRAFANA_PORT) || echo "Open http://localhost:$(GRAFANA_PORT) in your browser"

# Test the API with authentication
test-api:
	@echo "🧪 Testing API endpoints..."
	@./scripts/test-api.sh

# Test embedding server
test-embed:
	@echo "🧪 Testing embedding server..."
	@echo ""
	@echo "Health check:"
	@curl -s http://localhost:$(EMBEDDING_PORT)/health | jq .
	@echo ""
	@echo "Drift test (related):"
	@curl -s -X POST http://localhost:$(EMBEDDING_PORT)/drift \
		-H "Content-Type: application/json" \
		-d '{"anchor": "I want to book a hotel in Paris", "message": "What hotels are near the Eiffel Tower?"}' | jq .
	@echo ""
	@echo "Drift test (unrelated):"
	@curl -s -X POST http://localhost:$(EMBEDDING_PORT)/drift \
		-H "Content-Type: application/json" \
		-d '{"anchor": "I want to book a hotel in Paris", "message": "How do I fix a Python memory leak?"}' | jq .

# Test drift detection flow
test-drift:
	@echo "🧪 Running drift detection test..."
	@chmod +x ./scripts/test-drift.sh
	@./scripts/test-drift.sh

build:
	@echo "🔨 Building for production..."
	@npm run build

# Testing & Quality
test:
	@echo "🧪 Running tests..."
	@npm test

test-watch:
	@echo "👀 Running tests in watch mode..."
	@npm run test:watch

lint:
	@echo "🔍 Running ESLint..."
	@npm run lint

format:
	@echo "✨ Formatting code..."
	@npm run format

typecheck:
	@echo "📝 Type checking..."
	@npm run typecheck

# Database
db-migrate:
	@echo "🗄️  Running database migrations..."
	@npm run db:migrate

db-studio:
	@echo "🎨 Opening Prisma Studio..."
	@npm run db:studio

db-push:
	@echo "⬆️  Pushing schema changes..."
	@npm run db:push

db-reset:
	@echo "⚠️  WARNING: This will delete all data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		npx prisma migrate reset --force; \
	fi

# Generators
generate:
	@echo "🎨 Running service generator..."
	@npm run generate

dashboards:
	@echo "📊 Generating Grafana dashboards..."
	@npm run generate:dashboards

# Cleanup
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf dist node_modules/.cache

docker-clean:
	@echo "⚠️  WARNING: This will delete all Docker volumes!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose -f docker/docker-compose.yml down -v; \
	fi

# Quick all-in-one
all: setup up db-migrate
	@echo ""
	@echo "✅ All set! Run 'make dev' to start both servers"
