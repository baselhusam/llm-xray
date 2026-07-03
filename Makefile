# LLM X-Ray — common dev/ops shortcuts. Run `make` to list targets.
COMPOSE := docker compose

.DEFAULT_GOAL := help

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

## --- Docker stack ---

build: ## Build both Docker images
	$(COMPOSE) build

up: ## Build (if needed) and start the full stack in the background
	$(COMPOSE) up -d --build

down: ## Stop and remove the stack (keeps the model cache volume)
	$(COMPOSE) down

logs: ## Follow logs from both services
	$(COMPOSE) logs -f

ps: ## Show running services
	$(COMPOSE) ps

clean: ## Stop the stack and delete volumes (drops the GPT-2 model cache)
	$(COMPOSE) down -v

## --- Local dev (no Docker) ---

dev-backend: ## Run the backend with autoreload (needs backend/.venv)
	cd backend && .venv/bin/uvicorn app.main:app --reload

dev-frontend: ## Run the Next.js dev server
	cd frontend && npm run dev

.PHONY: help build up down logs ps clean dev-backend dev-frontend
