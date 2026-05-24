package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestKeyRotatorRotatesAfterConfiguredRequestsPerModel(t *testing.T) {
	cfg := Config{
		DefaultPool: ModelKeyPool{GoogleModel: "gemini-default", APIKeys: []string{"a", "b"}, RequestsPerAPIKey: 2},
		Models: map[string]ModelKeyPool{
			"gpt-fast": {GoogleModel: "gemini-1.5-flash", APIKeys: []string{"x", "y"}, RequestsPerAPIKey: 1},
		},
	}
	r, err := NewKeyRotator(cfg)
	if err != nil {
		t.Fatal(err)
	}

	got := []string{}
	for i := 0; i < 4; i++ {
		lease, err := r.Lease("gpt-fast")
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, lease.APIKey)
	}
	want := []string{"x", "y", "x", "y"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("rotation mismatch at %d: got %v want %v", i, got, want)
		}
	}

	got = got[:0]
	for i := 0; i < 5; i++ {
		lease, err := r.Lease("unknown-model")
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, lease.APIKey)
	}
	want = []string{"a", "a", "b", "b", "a"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("default rotation mismatch at %d: got %v want %v", i, got, want)
		}
	}
}

func TestChatCompletionUsesOpenAIShapeAndRotatesKeys(t *testing.T) {
	var mu sync.Mutex
	keys := []string{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		keys = append(keys, r.URL.Query().Get("key"))
		mu.Unlock()
		if !strings.Contains(r.URL.Path, "/gemini-test:generateContent") {
			t.Fatalf("unexpected Google path: %s", r.URL.Path)
		}
		var body GeminiRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("bad request body: %v", err)
		}
		if body.SystemInstruction == nil || body.SystemInstruction.Parts[0].Text != "be concise" {
			t.Fatalf("system instruction not converted: %#v", body.SystemInstruction)
		}
		if body.Contents[0].Role != "user" || body.Contents[0].Parts[0].Text != "hello" {
			t.Fatalf("message not converted: %#v", body.Contents)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"hi back"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":4,"totalTokenCount":7}}`))
	}))
	defer upstream.Close()

	proxy := newTestProxy(t, upstream.URL, ModelKeyPool{GoogleModel: "gemini-test", APIKeys: []string{"k1", "k2"}, RequestsPerAPIKey: 1})
	server := httptest.NewServer(proxy)
	defer server.Close()

	for i := 0; i < 2; i++ {
		resp, err := http.Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"gpt-test","messages":[{"role":"system","content":"be concise"},{"role":"user","content":"hello"}]}`))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("unexpected status: %d", resp.StatusCode)
		}
		var out ChatCompletionResponse
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			t.Fatal(err)
		}
		if out.Object != "chat.completion" || out.Choices[0].Message.Content != "hi back" || out.Usage.TotalTokens != 7 {
			t.Fatalf("bad OpenAI response: %#v", out)
		}
	}

	mu.Lock()
	defer mu.Unlock()
	if strings.Join(keys, ",") != "k1,k2" {
		t.Fatalf("keys did not rotate per request: %v", keys)
	}
}

func TestRetryableUpstreamErrorForcesRotation(t *testing.T) {
	var mu sync.Mutex
	keys := []string{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := r.URL.Query().Get("key")
		mu.Lock()
		keys = append(keys, key)
		mu.Unlock()
		if key == "bad" {
			http.Error(w, `{"error":{"code":429,"message":"quota"}}`, http.StatusTooManyRequests)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	}))
	defer upstream.Close()

	proxy := newTestProxy(t, upstream.URL, ModelKeyPool{GoogleModel: "gemini-test", APIKeys: []string{"bad", "good"}, RequestsPerAPIKey: 100})
	server := httptest.NewServer(proxy)
	defer server.Close()

	resp, err := http.Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"gpt-test","messages":[{"role":"user","content":"hello"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected retry success, got status %d", resp.StatusCode)
	}
	mu.Lock()
	defer mu.Unlock()
	if strings.Join(keys, ",") != "bad,good" {
		t.Fatalf("expected retry on next key, got %v", keys)
	}
}

func newTestProxy(t *testing.T, upstreamURL string, pool ModelKeyPool) http.Handler {
	t.Helper()
	cfg := Config{
		GoogleBaseURL:   upstreamURL,
		UpstreamTimeout: 5 * time.Second,
		DefaultPool:     ModelKeyPool{GoogleModel: "gemini-default", APIKeys: []string{"default"}, RequestsPerAPIKey: 100},
		Models:          map[string]ModelKeyPool{"gpt-test": pool},
	}
	rotator, err := NewKeyRotator(cfg)
	if err != nil {
		t.Fatal(err)
	}
	proxy := &ProxyServer{cfg: cfg, keys: rotator, client: &http.Client{Timeout: cfg.UpstreamTimeout}, started: time.Now()}
	mux := http.NewServeMux()
	proxy.RegisterRoutes(mux)
	return mux
}
