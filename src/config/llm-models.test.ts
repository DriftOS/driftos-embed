import { describe, it, expect } from 'vitest';
import {
  getModelConfig,
  getApiKey,
  getResponseFormat,
  MODEL_CONFIGS,
} from './llm-models';

describe('getModelConfig', () => {
  it('returns config for known Groq model', () => {
    const config = getModelConfig('llama-3.1-8b-instant');
    expect(config.id).toBe('llama-3.1-8b-instant');
    expect(config.provider).toBe('groq');
    expect(config.baseUrl).toBe('https://api.groq.com/openai/v1');
  });

  it('returns config for known OpenAI model', () => {
    const config = getModelConfig('gpt-4o');
    expect(config.id).toBe('gpt-4o');
    expect(config.provider).toBe('openai');
    expect(config.baseUrl).toBe('https://api.openai.com/v1');
  });

  it('returns config for known Anthropic model', () => {
    const config = getModelConfig('claude-sonnet-4-5');
    expect(config.id).toBe('claude-sonnet-4-5');
    expect(config.provider).toBe('anthropic');
    expect(config.baseUrl).toBe('https://api.anthropic.com/v1');
  });

  it('returns fallback for unknown model', () => {
    const config = getModelConfig('unknown-model-xyz');
    expect(config.id).toBe('unknown-model-xyz');
    expect(config.provider).toBe('groq');
    expect(config.description).toContain('Unknown model');
  });

  it('includes supportsJsonSchema correctly', () => {
    expect(getModelConfig('gpt-4o').supportsJsonSchema).toBe(true);
    expect(getModelConfig('llama-3.1-8b-instant').supportsJsonSchema).toBe(false);
    expect(getModelConfig('claude-haiku-4-5').supportsJsonSchema).toBe(false);
  });

  it('includes supportsTemperature for GPT-5 models', () => {
    expect(getModelConfig('gpt-5').supportsTemperature).toBe(false);
    expect(getModelConfig('gpt-5-mini').supportsTemperature).toBe(false);
    expect(getModelConfig('gpt-4o').supportsTemperature).toBeUndefined(); // defaults to true
  });
});

describe('getApiKey', () => {
  const mockEnv = {
    GROQ_API_KEY: 'groq-key-123',
    OPENAI_API_KEY: 'openai-key-456',
    ANTHROPIC_API_KEY: 'anthropic-key-789',
  } as any;

  it('returns Groq API key', () => {
    expect(getApiKey('groq', mockEnv)).toBe('groq-key-123');
  });

  it('returns OpenAI API key', () => {
    expect(getApiKey('openai', mockEnv)).toBe('openai-key-456');
  });

  it('returns Anthropic API key', () => {
    expect(getApiKey('anthropic', mockEnv)).toBe('anthropic-key-789');
  });

  it('throws when API key is missing', () => {
    const emptyEnv = {
      GROQ_API_KEY: '',
      OPENAI_API_KEY: '',
      ANTHROPIC_API_KEY: '',
    } as any;

    expect(() => getApiKey('groq', emptyEnv)).toThrow('No API key configured for provider: groq');
  });
});

describe('getResponseFormat', () => {
  it('returns undefined for Anthropic models', () => {
    expect(getResponseFormat('claude-sonnet-4-5')).toBeUndefined();
    expect(getResponseFormat('claude-haiku-4-5')).toBeUndefined();
    expect(getResponseFormat('claude-opus-4-5')).toBeUndefined();
  });

  it('returns json_object for models without schema support', () => {
    const result = getResponseFormat('llama-3.1-8b-instant');
    expect(result).toEqual({ type: 'json_object' });
  });

  it('returns json_object when no schema provided', () => {
    const result = getResponseFormat('gpt-4o');
    expect(result).toEqual({ type: 'json_object' });
  });

  it('returns json_schema when model supports it and schema provided', () => {
    const schema = { name: 'test', schema: { type: 'object' } };
    const result = getResponseFormat('gpt-4o', schema);

    expect(result).toEqual({
      type: 'json_schema',
      json_schema: schema,
    });
  });

  it('returns json_object for models without schema support even with schema', () => {
    const schema = { name: 'test', schema: { type: 'object' } };
    const result = getResponseFormat('llama-3.1-8b-instant', schema);

    expect(result).toEqual({ type: 'json_object' });
  });

  it('returns json_schema for Llama 4 models with schema', () => {
    const schema = { name: 'test', schema: { type: 'object' } };
    const result = getResponseFormat('meta-llama/llama-4-scout-17b-16e-instruct', schema);

    expect(result).toEqual({
      type: 'json_schema',
      json_schema: schema,
    });
  });
});

describe('MODEL_CONFIGS', () => {
  it('has all expected providers', () => {
    const providers = new Set(Object.values(MODEL_CONFIGS).map((c) => c.provider));
    expect(providers).toContain('groq');
    expect(providers).toContain('openai');
    expect(providers).toContain('anthropic');
  });

  it('all configs have required fields', () => {
    for (const [key, config] of Object.entries(MODEL_CONFIGS)) {
      expect(config.id).toBe(key);
      expect(config.provider).toBeDefined();
      expect(config.baseUrl).toBeDefined();
      expect(typeof config.supportsJsonSchema).toBe('boolean');
      expect(typeof config.defaultTemperature).toBe('number');
      expect(typeof config.maxTokens).toBe('number');
      expect(config.description).toBeDefined();
    }
  });
});
