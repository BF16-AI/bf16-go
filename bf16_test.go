package bf16

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChatCompletionAndMe(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer sk-bf16-test" {
			http.Error(w, `{"error":{"message":"unauthorized","code":"invalid_api_key"}}`, http.StatusUnauthorized)
			return
		}
		switch r.URL.Path {
		case "/chat/completions":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":    "chatcmpl-1",
				"model": ModelDeepSeekV41FlashFast,
				"choices": []map[string]any{
					{
						"index":         0,
						"message":       map[string]string{"role": "assistant", "content": "ok"},
						"finish_reason": "stop",
					},
				},
			})
		case "/me":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"balance": map[string]any{"available_usd": 12.5, "reserved_usd": 0.0},
				"tier":    map[string]any{"id": 1, "name": "Tier 1", "max_concurrent_requests": 10},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	client := NewClient("sk-bf16-test", WithBaseURL(srv.URL))
	resp, err := client.ChatCompletion(context.Background(), ChatCompletionRequest{
		Messages: []Message{{Role: "user", Content: "ping"}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion error: %v", err)
	}
	if len(resp.Choices) != 1 || resp.Choices[0].Message.Content != "ok" {
		t.Fatalf("unexpected response: %+v", resp)
	}

	me, err := client.Me(context.Background())
	if err != nil {
		t.Fatalf("Me error: %v", err)
	}
	if me.Balance.AvailableUSD != 12.5 || me.Tier.ID != 1 {
		t.Fatalf("unexpected account info: %+v", me)
	}
}

func TestChatCompletionStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello \"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"bf16\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	client := NewClient("sk-bf16-test", WithBaseURL(srv.URL))
	var out strings.Builder
	err := client.ChatCompletionStream(context.Background(), ChatCompletionRequest{
		Messages: []Message{{Role: "user", Content: "hi"}},
	}, func(chunk StreamChunk) error {
		if len(chunk.Choices) > 0 {
			out.WriteString(chunk.Choices[0].Delta.Content)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("ChatCompletionStream error: %v", err)
	}
	if out.String() != "Hello bf16" {
		t.Fatalf("got %q, want %q", out.String(), "Hello bf16")
	}
}
