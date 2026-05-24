package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type ProxyServer struct {
	cfg     Config
	keys    *KeyRotator
	client  *http.Client
	started time.Time
}

func (s *ProxyServer) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/healthz", s.healthz)
	mux.HandleFunc("/v1/models", s.auth(s.models))
	mux.HandleFunc("/models", s.auth(s.models))
	mux.HandleFunc("/v1/chat/completions", s.auth(s.chatCompletions))
	mux.HandleFunc("/chat/completions", s.auth(s.chatCompletions))
}

func (s *ProxyServer) auth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if len(s.cfg.ProxyAPIKeys) == 0 {
			next(w, r)
			return
		}
		tok := bearerToken(r.Header.Get("Authorization"))
		if _, ok := s.cfg.ProxyAPIKeys[tok]; !ok {
			writeError(w, http.StatusUnauthorized, "invalid or missing proxy API key", "invalid_request_error")
			return
		}
		next(w, r)
	}
}

func (s *ProxyServer) healthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "uptime_seconds": int(time.Since(s.started).Seconds())})
}

func (s *ProxyServer) models(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
		return
	}
	writeJSON(w, http.StatusOK, ModelListResponse{Object: "list", Data: s.keys.Models()})
}

func (s *ProxyServer) chatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
		return
	}
	defer r.Body.Close()
	var req ChatCompletionRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 20<<20)).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error(), "invalid_request_error")
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model is required", "invalid_request_error")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages are required", "invalid_request_error")
		return
	}

	geminiReq, err := toGeminiRequest(req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}

	if req.Stream {
		s.streamGemini(w, r, req, geminiReq)
		return
	}
	s.generateGemini(w, r, req, geminiReq)
}

func (s *ProxyServer) generateGemini(w http.ResponseWriter, r *http.Request, openReq ChatCompletionRequest, geminiReq GeminiRequest) {
	var lastErr string
	attempts := s.maxAttempts(openReq.Model)
	for i := 0; i < attempts; i++ {
		lease, err := s.keys.Lease(openReq.Model)
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
			return
		}
		body, status, err := s.callGoogle(r, lease, "generateContent", geminiReq)
		if err != nil {
			lastErr = err.Error()
			if retryableStatus(status) {
				s.keys.ForceRotate(openReq.Model, lease.Index)
				continue
			}
			writeError(w, statusOr(status, http.StatusBadGateway), lastErr, "upstream_error")
			return
		}
		var gr GeminiResponse
		if err := json.Unmarshal(body, &gr); err != nil {
			writeError(w, http.StatusBadGateway, "invalid Google response: "+err.Error(), "upstream_error")
			return
		}
		if gr.Error != nil {
			lastErr = gr.Error.Message
			if retryableStatus(gr.Error.Code) {
				s.keys.ForceRotate(openReq.Model, lease.Index)
				continue
			}
			writeError(w, googleHTTPStatus(gr.Error.Code), gr.Error.Message, "upstream_error")
			return
		}
		writeJSON(w, http.StatusOK, toOpenAIResponse(openReq, gr))
		return
	}
	writeError(w, http.StatusBadGateway, "all configured Google API keys failed or were rate-limited: "+lastErr, "upstream_error")
}

func (s *ProxyServer) streamGemini(w http.ResponseWriter, r *http.Request, openReq ChatCompletionRequest, geminiReq GeminiRequest) {
	lease, err := s.keys.Lease(openReq.Model)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	payload, _ := json.Marshal(geminiReq)
	endpoint := s.googleURL(lease.GoogleModel, "streamGenerateContent", lease.APIKey)
	endpoint += "&alt=sse"
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error(), "upstream_error")
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if retryableStatus(resp.StatusCode) {
			s.keys.ForceRotate(openReq.Model, lease.Index)
		}
		writeError(w, googleHTTPStatus(resp.StatusCode), string(b), "upstream_error")
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming unsupported by server", "server_error")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	id := newCompletionID()
	created := time.Now().Unix()
	writeSSE(w, ChatCompletionChunk{ID: id, Object: "chat.completion.chunk", Created: created, Model: openReq.Model, Choices: []ChatChoice{{Index: 0, Delta: &ChatDelta{Role: "assistant"}}}})
	flusher.Flush()

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 16*1024), 2*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var gr GeminiResponse
		if err := json.Unmarshal([]byte(data), &gr); err != nil {
			slog.Warn("skipping malformed Google stream event", "error", err)
			continue
		}
		finish := geminiFinishReason(gr)
		if len(gr.Candidates) == 0 {
			continue
		}
		for i, p := range gr.Candidates[0].Content.Parts {
			if p.FunctionCall != nil {
				argsJSON, _ := json.Marshal(p.FunctionCall.Args)
				delta := &ChatDelta{ToolCalls: []DeltaToolCall{{
					Index: i,
					ID:    fmt.Sprintf("call_%d", i),
					Type:  "function",
					Function: DeltaFunction{
						Name:      p.FunctionCall.Name,
						Arguments: string(argsJSON),
					},
				}}}
				writeSSE(w, ChatCompletionChunk{ID: id, Object: "chat.completion.chunk", Created: created, Model: openReq.Model, Choices: []ChatChoice{{Index: 0, Delta: delta, FinishReason: finish}}})
				flusher.Flush()
			} else if p.Text != "" {
				writeSSE(w, ChatCompletionChunk{ID: id, Object: "chat.completion.chunk", Created: created, Model: openReq.Model, Choices: []ChatChoice{{Index: 0, Delta: &ChatDelta{Content: p.Text}, FinishReason: finish}}})
				flusher.Flush()
			}
		}
	}
	writeRawSSE(w, "[DONE]")
	flusher.Flush()
}

func (s *ProxyServer) callGoogle(r *http.Request, lease KeyLease, action string, payload any) ([]byte, int, error) {
	b, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, s.googleURL(lease.GoogleModel, action, lease.APIKey), bytes.NewReader(b))
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, http.StatusBadGateway, err
	}
	defer resp.Body.Close()
	body, readErr := io.ReadAll(io.LimitReader(resp.Body, 20<<20))
	if readErr != nil {
		return nil, resp.StatusCode, readErr
	}
	if resp.StatusCode >= 400 {
		return body, resp.StatusCode, fmt.Errorf("Google API returned HTTP %d: %s", resp.StatusCode, string(body))
	}
	return body, resp.StatusCode, nil
}

func (s *ProxyServer) googleURL(model, action, apiKey string) string {
	escapedModel := url.PathEscape(model)
	return fmt.Sprintf("%s/%s:%s?key=%s", s.cfg.GoogleBaseURL, escapedModel, action, url.QueryEscape(apiKey))
}

func (s *ProxyServer) maxAttempts(model string) int {
	pool := s.keys.def
	if p := s.keys.pools[model]; p != nil {
		pool = p
	}
	if pool == nil || len(pool.keys) < 1 {
		return 1
	}
	return len(pool.keys)
}

func toGeminiRequest(req ChatCompletionRequest) (GeminiRequest, error) {
	out := GeminiRequest{GenerationConfig: &GenerationConfig{Temperature: req.Temperature, TopP: req.TopP, MaxOutputTokens: req.MaxTokens}}
	for _, m := range req.Messages {
		text := contentToText(m.Content)
		switch m.Role {
		case "system", "developer":
			if text != "" {
				if out.SystemInstruction == nil {
					out.SystemInstruction = &GeminiContent{Parts: []GeminiPart{{Text: text}}}
				} else {
					out.SystemInstruction.Parts = append(out.SystemInstruction.Parts, GeminiPart{Text: text})
				}
			}
		case "assistant":
			var parts []GeminiPart
			if text != "" {
				parts = append(parts, GeminiPart{Text: text})
			}
			for _, tc := range m.ToolCalls {
				var args any
				if tc.Function.Arguments != "" {
					_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
					if args == nil {
						args = tc.Function.Arguments
					}
				}
				parts = append(parts, GeminiPart{FunctionCall: &GeminiFunctionCall{Name: tc.Function.Name, Args: args}})
			}
			if len(parts) > 0 {
				out.Contents = append(out.Contents, GeminiContent{Role: "model", Parts: parts})
			}
		case "user":
			if text != "" {
				out.Contents = append(out.Contents, GeminiContent{Role: "user", Parts: []GeminiPart{{Text: text}}})
			}
		case "tool":
			var resp any
			if text != "" {
				_ = json.Unmarshal([]byte(text), &resp)
				if resp == nil {
					resp = map[string]any{"output": text}
				}
			} else {
				resp = map[string]any{}
			}
			out.Contents = append(out.Contents, GeminiContent{Role: "user", Parts: []GeminiPart{{
				FunctionResponse: &GeminiFunctionResponse{Name: m.Name, Response: resp},
			}}})
		default:
			return out, fmt.Errorf("unsupported message role %q", m.Role)
		}
	}
	if len(out.Contents) == 0 {
		return out, fmt.Errorf("at least one non-system message with text content is required")
	}
	if len(req.Tools) > 0 {
		out.Tools = convertTools(req.Tools)
	}
	return out, nil
}

func contentToText(content any) string {
	switch v := content.(type) {
	case nil:
		return ""
	case string:
		return v
	case []any:
		parts := make([]string, 0, len(v))
		for _, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if t, _ := m["type"].(string); t == "text" {
				if text, _ := m["text"].(string); text != "" {
					parts = append(parts, text)
				}
			}
		}
		return strings.Join(parts, "\n")
	default:
		b, _ := json.Marshal(v)
		return string(b)
	}
}

func convertTools(tools []any) []GeminiTool {
	decls := []GeminiFunctionDeclaration{}
	for _, raw := range tools {
		m, ok := raw.(map[string]any)
		if !ok || m["type"] != "function" {
			continue
		}
		fn, ok := m["function"].(map[string]any)
		if !ok {
			continue
		}
		name, _ := fn["name"].(string)
		if name == "" {
			continue
		}
		desc, _ := fn["description"].(string)
		decls = append(decls, GeminiFunctionDeclaration{Name: name, Description: desc, Parameters: fn["parameters"]})
	}
	if len(decls) == 0 {
		return nil
	}
	return []GeminiTool{{FunctionDeclarations: decls}}
}

func toOpenAIResponse(req ChatCompletionRequest, gr GeminiResponse) ChatCompletionResponse {
	msg := geminiToMessage(gr)
	return ChatCompletionResponse{
		ID:      newCompletionID(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []ChatChoice{{Index: 0, Message: msg, FinishReason: geminiFinishReason(gr)}},
		Usage:   Usage{PromptTokens: gr.UsageMetadata.PromptTokenCount, CompletionTokens: gr.UsageMetadata.CandidatesTokenCount, TotalTokens: gr.UsageMetadata.TotalTokenCount},
	}
}

// geminiToMessage converts a GeminiResponse candidate into an OpenAI ChatMessage,
// including text content and any functionCall parts mapped to tool_calls.
func geminiToMessage(gr GeminiResponse) ChatMessage {
	if len(gr.Candidates) == 0 {
		return ChatMessage{Role: "assistant"}
	}
	parts := gr.Candidates[0].Content.Parts
	msg := ChatMessage{Role: "assistant"}
	var texts []string
	for i, p := range parts {
		if p.FunctionCall != nil {
			argsJSON, _ := json.Marshal(p.FunctionCall.Args)
			msg.ToolCalls = append(msg.ToolCalls, ToolCall{
				ID:   fmt.Sprintf("call_%d", i),
				Type: "function",
				Function: FunctionCall{
					Name:      p.FunctionCall.Name,
					Arguments: string(argsJSON),
				},
			})
		} else if p.Text != "" {
			texts = append(texts, p.Text)
		}
	}
	if len(texts) > 0 {
		msg.Content = strings.Join(texts, "")
	}
	return msg
}

func geminiText(gr GeminiResponse) string {
	if len(gr.Candidates) == 0 {
		return ""
	}
	parts := gr.Candidates[0].Content.Parts
	texts := make([]string, 0, len(parts))
	for _, p := range parts {
		if p.Text != "" {
			texts = append(texts, p.Text)
		}
	}
	return strings.Join(texts, "")
}

func geminiFinishReason(gr GeminiResponse) string {
	if len(gr.Candidates) == 0 {
		return ""
	}
	switch gr.Candidates[0].FinishReason {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY", "RECITATION", "OTHER":
		return "content_filter"
	default:
		return ""
	}
}

func bearerToken(header string) string {
	if strings.HasPrefix(strings.ToLower(header), "bearer ") {
		return strings.TrimSpace(header[7:])
	}
	return ""
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, message, typ string) {
	if status == 0 {
		status = http.StatusInternalServerError
	}
	writeJSON(w, status, ErrorResponse{Error: OpenAIError{Message: message, Type: typ}})
}

func writeSSE(w io.Writer, v any) {
	b, _ := json.Marshal(v)
	writeRawSSE(w, string(b))
}

func writeRawSSE(w io.Writer, data string) { _, _ = fmt.Fprintf(w, "data: %s\n\n", data) }

func retryableStatus(status int) bool {
	return status == 429 || status == 500 || status == 502 || status == 503 || status == 504
}
func statusOr(status, fallback int) int {
	if status == 0 {
		return fallback
	}
	return googleHTTPStatus(status)
}
func googleHTTPStatus(status int) int {
	if status >= 400 && status <= 599 {
		return status
	}
	return http.StatusBadGateway
}
