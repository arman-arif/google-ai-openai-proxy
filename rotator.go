package main

import (
	"fmt"
	"sync"
)

type KeyRotator struct {
	mu    sync.Mutex
	pools map[string]*runtimePool
	def   *runtimePool
}

type runtimePool struct {
	openAIModel string
	googleModel string
	keys        []string
	limit       int64
	idx         int
	used        int64
}

type KeyLease struct {
	OpenAIModel string
	GoogleModel string
	APIKey      string
	Index       int
}

type KeyFailureAction int

const (
	KeyFailureNone KeyFailureAction = iota
	KeyFailureRotate
	KeyFailureRemove
)

func NewKeyRotator(cfg Config) (*KeyRotator, error) {
	def, err := buildPool("*", cfg.DefaultPool)
	if err != nil && len(cfg.DefaultPool.APIKeys) > 0 {
		return nil, err
	}
	r := &KeyRotator{pools: map[string]*runtimePool{}, def: def}
	for model, poolCfg := range cfg.Models {
		if len(poolCfg.APIKeys) == 0 {
			poolCfg.APIKeys = cfg.DefaultPool.APIKeys
		}
		if poolCfg.RequestsPerAPIKey == 0 {
			poolCfg.RequestsPerAPIKey = cfg.DefaultPool.RequestsPerAPIKey
		}
		if poolCfg.GoogleModel == "" {
			poolCfg.GoogleModel = model
		}
		pool, err := buildPool(model, poolCfg)
		if err != nil {
			return nil, err
		}
		r.pools[model] = pool
	}
	if r.def == nil && len(r.pools) == 0 {
		return nil, fmt.Errorf("no usable key pools configured")
	}
	return r, nil
}

func buildPool(openAIModel string, cfg ModelKeyPool) (*runtimePool, error) {
	if len(cfg.APIKeys) == 0 {
		return nil, fmt.Errorf("model %q has no api_keys", openAIModel)
	}
	if cfg.RequestsPerAPIKey < 1 {
		return nil, fmt.Errorf("model %q requests_per_api_key must be >= 1", openAIModel)
	}
	if cfg.GoogleModel == "" || cfg.GoogleModel == "*" {
		cfg.GoogleModel = openAIModel
	}
	return &runtimePool{openAIModel: openAIModel, googleModel: cfg.GoogleModel, keys: cfg.APIKeys, limit: cfg.RequestsPerAPIKey}, nil
}

func (r *KeyRotator) Lease(model string) (KeyLease, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	pool := r.pools[model]
	if pool == nil {
		pool = r.def
	}
	if pool == nil || len(pool.keys) == 0 {
		return KeyLease{}, fmt.Errorf("no key pool configured for model %q", model)
	}

	lease := KeyLease{OpenAIModel: model, GoogleModel: pool.googleModel, APIKey: pool.keys[pool.idx], Index: pool.idx}
	pool.used++
	if pool.used >= pool.limit {
		pool.used = 0
		pool.idx = (pool.idx + 1) % len(pool.keys)
	}
	return lease, nil
}

func (r *KeyRotator) ForceRotate(model string, keyIndex int) {
	r.markKeyFailed(model, keyIndex, KeyFailureRotate)
}

func (r *KeyRotator) RemoveKey(model string, keyIndex int) {
	r.markKeyFailed(model, keyIndex, KeyFailureRemove)
}

func (r *KeyRotator) markKeyFailed(model string, keyIndex int, action KeyFailureAction) {
	r.mu.Lock()
	defer r.mu.Unlock()
	pool := r.pools[model]
	if pool == nil {
		pool = r.def
	}
	if pool == nil || len(pool.keys) == 0 || pool.idx != keyIndex {
		return
	}
	pool.used = 0
	if action == KeyFailureRemove && len(pool.keys) > 1 {
		pool.keys = append(pool.keys[:keyIndex], pool.keys[keyIndex+1:]...)
		if keyIndex >= len(pool.keys) {
			pool.idx = 0
		} else {
			pool.idx = keyIndex
		}
		return
	}
	pool.idx = (pool.idx + 1) % len(pool.keys)
}

func (r *KeyRotator) Models() []ModelInfo {
	r.mu.Lock()
	defer r.mu.Unlock()
	models := make([]ModelInfo, 0, len(r.pools)+1)
	for name := range r.pools {
		models = append(models, ModelInfo{ID: name, Object: "model", OwnedBy: "google-ai-proxy"})
	}
	if len(models) == 0 && r.def != nil {
		models = append(models, ModelInfo{ID: r.def.googleModel, Object: "model", OwnedBy: "google-ai-proxy"})
	}
	return models
}
