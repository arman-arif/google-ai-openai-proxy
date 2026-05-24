package main

import (
	"bufio"
	"encoding/json"
	"fmt"
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

func TestToolCallRoundtrip(t *testing.T) {
	// Simulates a Gemini response that contains a functionCall part.
	// Verifies the proxy translates it back into OpenAI tool_calls shape.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify tools were forwarded as Gemini functionDeclarations
		var body GeminiRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("bad request body: %v", err)
		}
		if len(body.Tools) == 0 || len(body.Tools[0].FunctionDeclarations) == 0 {
			t.Fatalf("expected functionDeclarations, got: %#v", body.Tools)
		}
		if body.Tools[0].FunctionDeclarations[0].Name != "get_weather" {
			t.Fatalf("unexpected function name: %s", body.Tools[0].FunctionDeclarations[0].Name)
		}
		// Respond with a functionCall part
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"candidates":[{
				"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"London"}}}]},
				"finishReason":"STOP"
			}]
		}`))
	}))
	defer upstream.Close()

	proxy := newTestProxy(t, upstream.URL, ModelKeyPool{GoogleModel: "gemini-test", APIKeys: []string{"k1"}, RequestsPerAPIKey: 100})
	server := httptest.NewServer(proxy)
	defer server.Close()

	reqBody := `{
		"model":"gpt-test",
		"messages":[{"role":"user","content":"What is the weather in London?"}],
		"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}]
	}`
	resp, err := http.Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	var out ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if len(out.Choices) == 0 {
		t.Fatal("no choices in response")
	}
	msg := out.Choices[0].Message
	if len(msg.ToolCalls) == 0 {
		t.Fatalf("expected tool_calls in response, got none. message: %#v", msg)
	}
	tc := msg.ToolCalls[0]
	if tc.Type != "function" {
		t.Fatalf("expected type=function, got %q", tc.Type)
	}
	if tc.Function.Name != "get_weather" {
		t.Fatalf("expected name=get_weather, got %q", tc.Function.Name)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		t.Fatalf("tool call arguments not valid JSON: %v", err)
	}
	if args["location"] != "London" {
		t.Fatalf("expected location=London, got %v", args["location"])
	}
}

func TestToolResultForwardedAsFunctionResponse(t *testing.T) {
	// Verifies that a tool role message is forwarded as a Gemini functionResponse part.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body GeminiRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("bad body: %v", err)
		}
		// Expect: user(hello) → model(functionCall) → user(functionResponse)
		if len(body.Contents) < 3 {
			t.Fatalf("expected 3 content turns, got %d: %#v", len(body.Contents), body.Contents)
		}
		lastTurn := body.Contents[2]
		if lastTurn.Role != "user" {
			t.Fatalf("expected last role=user, got %q", lastTurn.Role)
		}
		if len(lastTurn.Parts) == 0 || lastTurn.Parts[0].FunctionResponse == nil {
			t.Fatalf("expected functionResponse part, got: %#v", lastTurn.Parts)
		}
		if lastTurn.Parts[0].FunctionResponse.Name != "get_weather" {
			t.Fatalf("expected functionResponse.name=get_weather, got %q", lastTurn.Parts[0].FunctionResponse.Name)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"It is 15°C in London."}]},"finishReason":"STOP"}]}`))
	}))
	defer upstream.Close()

	proxy := newTestProxy(t, upstream.URL, ModelKeyPool{GoogleModel: "gemini-test", APIKeys: []string{"k1"}, RequestsPerAPIKey: 100})
	server := httptest.NewServer(proxy)
	defer server.Close()

	reqBody := `{
		"model":"gpt-test",
		"messages":[
			{"role":"user","content":"What is the weather in London?"},
			{"role":"assistant","tool_calls":[{"id":"call_0","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"London\"}"}}]},
			{"role":"tool","tool_call_id":"call_0","name":"get_weather","content":"{\"temperature\":15,\"unit\":\"C\"}"}
		]
	}`
	resp, err := http.Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	var out ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Choices[0].Message.Content != "It is 15°C in London." {
		t.Fatalf("unexpected content: %q", out.Choices[0].Message.Content)
	}
}


func TestStreamingRetriesOnRateLimitBeforeHeadersWritten(t *testing.T) {
	var mu sync.Mutex
	keys := []string{}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := r.URL.Query().Get("key")
		mu.Lock()
		keys = append(keys, key)
		mu.Unlock()

		if key == "bad" {
			http.Error(w, `{"error":{"code":429,"message":"quota exceeded"}}`, http.StatusTooManyRequests)
			return
		}
		// Good key: respond with a minimal SSE stream
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hello\"}]},\"finishReason\":\"STOP\"}]}\n\n")
	}))
	defer upstream.Close()

	proxy := newTestProxy(t, upstream.URL, ModelKeyPool{
		GoogleModel:       "gemini-test",
		APIKeys:           []string{"bad", "good"},
		RequestsPerAPIKey: 100,
	})
	server := httptest.NewServer(proxy)
	defer server.Close()

	resp, err := http.Post(server.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"gpt-test","messages":[{"role":"user","content":"hi"}],"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("expected SSE content type, got %q", ct)
	}

	// Read chunks and collect text
	var text string
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var chunk ChatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta != nil {
			text += chunk.Choices[0].Delta.Content
		}
	}

	mu.Lock()
	defer mu.Unlock()
	if strings.Join(keys, ",") != "bad,good" {
		t.Fatalf("expected retry on good key after bad 429, got keys: %v", keys)
	}
	if text != "hello" {
		t.Fatalf("expected streamed text 'hello', got %q", text)
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
