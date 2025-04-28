import { Anthropic } from "@anthropic-ai/sdk"
import { AnthropicVertex } from "@anthropic-ai/vertex-sdk"
import { Stream as AnthropicStream } from "@anthropic-ai/sdk/streaming"
import { GoogleAuth } from "google-auth-library"

import { ApiHandlerOptions, ModelInfo, vertexDefaultModelId, VertexModelId, vertexModels } from "../../shared/api"
import { ApiStream } from "../transform/stream"

import { getModelParams, SingleCompletionHandler } from "../index"
import { BaseProvider } from "./base-provider"
import { ANTHROPIC_DEFAULT_MAX_TOKENS } from "./constants"
import { formatMessageForCache } from "../transform/vertex-caching"

interface VertexUsage {
	input_tokens?: number
	output_tokens?: number
	cache_creation_input_tokens?: number
	cache_read_input_tokens?: number
}

interface VertexMessageResponse {
	content: Array<{ type: "text"; text: string }>
}

interface VertexMessageStreamEvent {
	type: "message_start" | "message_delta" | "content_block_start" | "content_block_delta"
	message?: {
		usage: VertexUsage
	}
	usage?: {
		output_tokens: number
	}
	content_block?: { type: "text"; text: string } | { type: "thinking"; thinking: string }
	index?: number
	delta?: { type: "text_delta"; text: string } | { type: "thinking_delta"; thinking: string }
}

// https://docs.anthropic.com/en/api/claude-on-vertex-ai
export class AnthropicVertexHandler extends BaseProvider implements SingleCompletionHandler {
	protected options: ApiHandlerOptions
	private client: AnthropicVertex

	constructor(options: ApiHandlerOptions) {
		super()

		this.options = options

		if (this.options.vertexJsonCredentials) {
			this.client = new AnthropicVertex({
				projectId: this.options.vertexProjectId ?? "not-provided",
				// https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude#regions
				region: this.options.vertexRegion ?? "us-east5",
				googleAuth: new GoogleAuth({
					scopes: ["https://www.googleapis.com/auth/cloud-platform"],
					credentials: JSON.parse(this.options.vertexJsonCredentials),
				}),
			})
		} else if (this.options.vertexKeyFile) {
			this.client = new AnthropicVertex({
				projectId: this.options.vertexProjectId ?? "not-provided",
				// https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude#regions
				region: this.options.vertexRegion ?? "us-east5",
				googleAuth: new GoogleAuth({
					scopes: ["https://www.googleapis.com/auth/cloud-platform"],
					keyFile: this.options.vertexKeyFile,
				}),
			})
		} else {
			this.client = new AnthropicVertex({
				projectId: this.options.vertexProjectId ?? "not-provided",
				// https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude#regions
				region: this.options.vertexRegion ?? "us-east5",
			})
		}
	}

	override async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		const model = this.getModel()
		let { id, temperature, maxTokens, thinking } = model
		const useCache = model.info.supportsPromptCache

		// Find indices of user messages that we want to cache
		// We only cache the last two user messages to stay within the 4-block limit
		// (1 block for system + 1 block each for last two user messages = 3 total)
		const userMsgIndices = useCache
			? messages.reduce((acc, msg, i) => (msg.role === "user" ? [...acc, i] : acc), [] as number[])
			: []

		const lastUserMsgIndex = userMsgIndices[userMsgIndices.length - 1] ?? -1
		const secondLastMsgUserIndex = userMsgIndices[userMsgIndices.length - 2] ?? -1

		/**
		 * Vertex API has specific limitations for prompt caching:
		 * 1. Maximum of 4 blocks can have cache_control
		 * 2. Only text blocks can be cached (images and other content types cannot)
		 * 3. Cache control can only be applied to user messages, not assistant messages
		 *
		 * Our caching strategy:
		 * - Cache the system prompt (1 block)
		 * - Cache the last text block of the second-to-last user message (1 block)
		 * - Cache the last text block of the last user message (1 block)
		 * This ensures we stay under the 4-block limit while maintaining effective caching
		 * for the most relevant context.
		 */
		const params = {
			model: id,
			max_tokens: maxTokens,
			temperature,
			thinking,
			// Cache the system prompt if caching is enabled.
			system: useCache
				? [{ text: systemPrompt, type: "text" as const, cache_control: { type: "ephemeral" } }]
				: systemPrompt,
			messages: messages.map((message, index) => {
				// Only cache the last two user messages.
				const shouldCache = useCache && (index === lastUserMsgIndex || index === secondLastMsgUserIndex)
				return formatMessageForCache(message, shouldCache)
			}),
			stream: true,
		}

		const stream = (await this.client.messages.create(
			params as Anthropic.Messages.MessageCreateParamsStreaming,
		)) as unknown as AnthropicStream<VertexMessageStreamEvent>

		for await (const chunk of stream) {
			switch (chunk.type) {
				case "message_start": {
					const usage = chunk.message!.usage

					yield {
						type: "usage",
						inputTokens: usage.input_tokens || 0,
						outputTokens: usage.output_tokens || 0,
						cacheWriteTokens: usage.cache_creation_input_tokens,
						cacheReadTokens: usage.cache_read_input_tokens,
					}

					break
				}
				case "message_delta": {
					yield {
						type: "usage",
						inputTokens: 0,
						outputTokens: chunk.usage!.output_tokens || 0,
					}

					break
				}
				case "content_block_start": {
					switch (chunk.content_block!.type) {
						case "text": {
							if (chunk.index! > 0) {
								yield { type: "text", text: "\n" }
							}

							yield { type: "text", text: chunk.content_block!.text }
							break
						}
						case "thinking": {
							if (chunk.index! > 0) {
								yield { type: "reasoning", text: "\n" }
							}

							yield { type: "reasoning", text: (chunk.content_block as any).thinking }
							break
						}
					}
					break
				}
				case "content_block_delta": {
					switch (chunk.delta!.type) {
						case "text_delta": {
							yield { type: "text", text: chunk.delta!.text }
							break
						}
						case "thinking_delta": {
							yield { type: "reasoning", text: (chunk.delta as any).thinking }
							break
						}
					}
					break
				}
			}
		}
	}

	getModel() {
		const modelId = this.options.apiModelId
		let id = modelId && modelId in vertexModels ? (modelId as VertexModelId) : vertexDefaultModelId
		const info: ModelInfo = vertexModels[id]

		// The `:thinking` variant is a virtual identifier for thinking-enabled
		// models (similar to how it's handled in the Anthropic provider.)
		if (id.endsWith(":thinking")) {
			id = id.replace(":thinking", "") as VertexModelId
		}

		return {
			id,
			info,
			...getModelParams({ options: this.options, model: info, defaultMaxTokens: ANTHROPIC_DEFAULT_MAX_TOKENS }),
		}
	}

	async completePrompt(prompt: string) {
		try {
			let { id, info, temperature, maxTokens, thinking } = this.getModel()
			const useCache = info.supportsPromptCache

			const params: Anthropic.Messages.MessageCreateParamsNonStreaming = {
				model: id,
				max_tokens: maxTokens ?? ANTHROPIC_DEFAULT_MAX_TOKENS,
				temperature,
				thinking,
				system: "", // No system prompt needed for single completions.
				messages: [
					{
						role: "user",
						content: useCache
							? [{ type: "text" as const, text: prompt, cache_control: { type: "ephemeral" } }]
							: prompt,
					},
				],
				stream: false,
			}

			const response = (await this.client.messages.create(params)) as unknown as VertexMessageResponse
			const content = response.content[0]

			if (content.type === "text") {
				return content.text
			}

			return ""
		} catch (error) {
			if (error instanceof Error) {
				throw new Error(`Vertex completion error: ${error.message}`)
			}

			throw error
		}
	}
}
