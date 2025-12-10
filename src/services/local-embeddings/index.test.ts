import { describe, it, expect } from 'vitest';
import { cosineSimilarity, getDriftAction, DRIFT_THRESHOLDS } from './index';

describe('cosineSimilarity', () => {
  it('returns 1 for identical vectors', () => {
    const v = [0.5, 0.5, 0.5];
    expect(cosineSimilarity(v, v)).toBeCloseTo(1.0);
  });

  it('returns 0 for orthogonal vectors', () => {
    const a = [1, 0];
    const b = [0, 1];
    expect(cosineSimilarity(a, b)).toBeCloseTo(0.0);
  });

  it('returns -1 for opposite vectors', () => {
    const a = [1, 0, 0];
    const b = [-1, 0, 0];
    expect(cosineSimilarity(a, b)).toBeCloseTo(-1.0);
  });

  it('handles normalized vectors correctly', () => {
    // Two normalized vectors at 60 degrees apart -> cos(60) = 0.5
    const a = [1, 0];
    const b = [0.5, Math.sqrt(3) / 2];
    expect(cosineSimilarity(a, b)).toBeCloseTo(0.5);
  });

  it('throws on dimension mismatch', () => {
    const a = [1, 2, 3];
    const b = [1, 2];
    expect(() => cosineSimilarity(a, b)).toThrow('Embedding dimension mismatch: 3 vs 2');
  });

  it('returns 0 for zero vectors', () => {
    const a = [0, 0, 0];
    const b = [1, 2, 3];
    expect(cosineSimilarity(a, b)).toBe(0);
  });

  it('handles high-dimensional vectors', () => {
    const dim = 384; // typical embedding dimension
    const a = new Array(dim).fill(1);
    const b = new Array(dim).fill(1);
    expect(cosineSimilarity(a, b)).toBeCloseTo(1.0);
  });

  it('is symmetric', () => {
    const a = [0.1, 0.5, 0.9];
    const b = [0.3, 0.6, 0.2];
    expect(cosineSimilarity(a, b)).toBeCloseTo(cosineSimilarity(b, a));
  });

  it('handles negative values', () => {
    const a = [-0.5, 0.5];
    const b = [0.5, 0.5];
    expect(cosineSimilarity(a, b)).toBeCloseTo(0.0);
  });
});

describe('getDriftAction', () => {
  it('returns STAY for high similarity (above stay threshold)', () => {
    expect(getDriftAction(0.6)).toBe('STAY');
    expect(getDriftAction(0.48)).toBe('STAY');
    expect(getDriftAction(0.9)).toBe('STAY');
  });

  it('returns BRANCH_SAME_CLUSTER for medium similarity', () => {
    expect(getDriftAction(0.3)).toBe('BRANCH_SAME_CLUSTER');
    expect(getDriftAction(0.4)).toBe('BRANCH_SAME_CLUSTER');
    expect(getDriftAction(0.21)).toBe('BRANCH_SAME_CLUSTER');
  });

  it('returns BRANCH_NEW_CLUSTER for low similarity', () => {
    expect(getDriftAction(0.1)).toBe('BRANCH_NEW_CLUSTER');
    expect(getDriftAction(0.0)).toBe('BRANCH_NEW_CLUSTER');
    expect(getDriftAction(0.19)).toBe('BRANCH_NEW_CLUSTER');
  });

  it('uses default thresholds correctly', () => {
    // Default: stay > 0.47, branch > 0.20
    expect(getDriftAction(0.47)).toBe('BRANCH_SAME_CLUSTER'); // exactly at threshold
    expect(getDriftAction(0.471)).toBe('STAY');
    expect(getDriftAction(0.20)).toBe('BRANCH_NEW_CLUSTER'); // exactly at threshold
    expect(getDriftAction(0.201)).toBe('BRANCH_SAME_CLUSTER');
  });

  it('respects custom thresholds', () => {
    // Custom: stay > 0.8, branch > 0.5
    expect(getDriftAction(0.9, 0.8, 0.5)).toBe('STAY');
    expect(getDriftAction(0.7, 0.8, 0.5)).toBe('BRANCH_SAME_CLUSTER');
    expect(getDriftAction(0.4, 0.8, 0.5)).toBe('BRANCH_NEW_CLUSTER');
  });

  it('handles edge cases at boundaries', () => {
    expect(getDriftAction(DRIFT_THRESHOLDS.stay)).toBe('BRANCH_SAME_CLUSTER');
    expect(getDriftAction(DRIFT_THRESHOLDS.branch)).toBe('BRANCH_NEW_CLUSTER');
  });

  it('handles negative similarity', () => {
    expect(getDriftAction(-0.5)).toBe('BRANCH_NEW_CLUSTER');
  });
});

describe('DRIFT_THRESHOLDS', () => {
  it('exports correct default values', () => {
    expect(DRIFT_THRESHOLDS.stay).toBe(0.47);
    expect(DRIFT_THRESHOLDS.branch).toBe(0.20);
  });
});
