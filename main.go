package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"
)

func main() {
	cfg, err := LoadConfig()
	if err != nil {
		slog.Error("configuration error", "error", err)
		os.Exit(1)
	}

	rotator, err := NewKeyRotator(cfg)
	if err != nil {
		slog.Error("key rotation setup failed", "error", err)
		os.Exit(1)
	}

	proxy := &ProxyServer{
		cfg:     cfg,
		keys:    rotator,
		client:  &http.Client{Timeout: cfg.UpstreamTimeout},
		started: time.Now(),
	}

	mux := http.NewServeMux()
	proxy.RegisterRoutes(mux)

	srv := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           logRequests(mux),
		ReadHeaderTimeout: 15 * time.Second,
	}

	go func() {
		slog.Info("google ai openai-compatible proxy listening", "addr", cfg.ListenAddr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("server failed", "error", err)
			os.Exit(1)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		slog.Error("graceful shutdown failed", "error", err)
		os.Exit(1)
	}
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rw, r)
		slog.Info("request", "method", r.Method, "path", r.URL.Path, "status", rw.status, "duration_ms", time.Since(start).Milliseconds())
	})
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(status int) {
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

// Config is intentionally stdlib-only: env vars are enough for containers,
// while optional JSON config supports per-model key pools and limits.
type Config struct {
	ListenAddr      string
	GoogleBaseURL   string
	UpstreamTimeout time.Duration
	ProxyAPIKeys    map[string]struct{}
	DefaultPool     ModelKeyPool
	Models          map[string]ModelKeyPool
}

type ModelKeyPool struct {
	GoogleModel       string   `json:"google_model"`
	APIKeys           []string `json:"api_keys"`
	RequestsPerAPIKey int64    `json:"requests_per_api_key"`
}

type fileConfig struct {
	ListenAddr      string                  `json:"listen_addr"`
	GoogleBaseURL   string                  `json:"google_base_url"`
	UpstreamTimeout string                  `json:"upstream_timeout"`
	ProxyAPIKeys    []string                `json:"proxy_api_keys"`
	Default         ModelKeyPool            `json:"default"`
	Models          map[string]ModelKeyPool `json:"models"`
}

func LoadConfig() (Config, error) {
	cfg := Config{
		ListenAddr:      env("LISTEN_ADDR", ":8080"),
		GoogleBaseURL:   strings.TrimRight(env("GOOGLE_API_BASE", "https://generativelanguage.googleapis.com/v1beta/models"), "/"),
		UpstreamTimeout: 120 * time.Second,
		ProxyAPIKeys:    map[string]struct{}{},
		Models:          map[string]ModelKeyPool{},
	}

	if d := os.Getenv("UPSTREAM_TIMEOUT"); d != "" {
		parsed, err := time.ParseDuration(d)
		if err != nil {
			return cfg, fmt.Errorf("invalid UPSTREAM_TIMEOUT: %w", err)
		}
		cfg.UpstreamTimeout = parsed
	}

	if path := os.Getenv("CONFIG_FILE"); path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			return cfg, fmt.Errorf("read CONFIG_FILE: %w", err)
		}
		var fc fileConfig
		if err := json.Unmarshal(b, &fc); err != nil {
			return cfg, fmt.Errorf("parse CONFIG_FILE: %w", err)
		}
		if fc.ListenAddr != "" {
			cfg.ListenAddr = fc.ListenAddr
		}
		if fc.GoogleBaseURL != "" {
			cfg.GoogleBaseURL = strings.TrimRight(fc.GoogleBaseURL, "/")
		}
		if fc.UpstreamTimeout != "" {
			parsed, err := time.ParseDuration(fc.UpstreamTimeout)
			if err != nil {
				return cfg, fmt.Errorf("invalid upstream_timeout: %w", err)
			}
			cfg.UpstreamTimeout = parsed
		}
		cfg.DefaultPool = fc.Default
		for _, k := range fc.ProxyAPIKeys {
			cfg.ProxyAPIKeys[k] = struct{}{}
		}
		for model, pool := range fc.Models {
			cfg.Models[model] = pool
		}
	}

	for _, k := range splitCSV(os.Getenv("PROXY_API_KEYS")) {
		cfg.ProxyAPIKeys[k] = struct{}{}
	}

	if len(cfg.DefaultPool.APIKeys) == 0 {
		cfg.DefaultPool.APIKeys = splitCSV(os.Getenv("GOOGLE_API_KEYS"))
	}
	if cfg.DefaultPool.RequestsPerAPIKey == 0 {
		cfg.DefaultPool.RequestsPerAPIKey = int64(envInt("REQUESTS_PER_API_KEY", 1000))
	}
	if cfg.DefaultPool.GoogleModel == "" {
		cfg.DefaultPool.GoogleModel = "gemini-1.5-flash"
	}
	if cfg.DefaultPool.RequestsPerAPIKey < 1 {
		return cfg, fmt.Errorf("requests_per_api_key must be >= 1")
	}
	if len(cfg.DefaultPool.APIKeys) == 0 && len(cfg.Models) == 0 {
		return cfg, fmt.Errorf("set GOOGLE_API_KEYS or CONFIG_FILE with at least one model key pool")
	}

	// MODEL_ALIASES='openai-name=google-model,other=gemini-1.5-pro' is a quick env-only path.
	for _, pair := range splitCSV(os.Getenv("MODEL_ALIASES")) {
		left, right, ok := strings.Cut(pair, "=")
		if !ok || strings.TrimSpace(left) == "" || strings.TrimSpace(right) == "" {
			return cfg, fmt.Errorf("invalid MODEL_ALIASES entry %q, expected openai=google", pair)
		}
		pool := cfg.DefaultPool
		pool.GoogleModel = strings.TrimSpace(right)
		cfg.Models[strings.TrimSpace(left)] = pool
	}

	return cfg, nil
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

func splitCSV(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}
